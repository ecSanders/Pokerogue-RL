/*
 * Headless PokeRogue simulator for RL training.
 *
 * Runs inside the Vitest environment (which provides the jsdom/Phaser mocks the game needs)
 * and serves newline-delimited JSON over a localhost TCP socket:
 *
 *   -> {"cmd":"reset","seed":"abc","mode":"endless","starters":["CHARMANDER"],"maxWave":200}
 *   -> {"cmd":"step","action":3}
 *   -> {"cmd":"close"}      (ends this connection; the server keeps listening)
 *   -> {"cmd":"shutdown"}   (stops the server)
 *   <- {"ok":true,"obs":{...},"done":false,"info":{...}}  |  {"ok":false,"error":"..."}
 *
 * Every observation has a decision `kind` and a `legal` mask over that kind's actions:
 *   command        0-3 = use move slot, 4+k = switch to party slot k, 10+b = throw ball type b
 *   reward         index into obs.options (rewards, shop items, reroll, skip)
 *   reward_target  party slot to receive the chosen item
 *   learn_move     0-3 = replace that move slot with obs.learn.move, 4 = don't learn
 *   target         battler index to aim a single-target move at in doubles: 0-1 our side, 2-3 enemies
 *   faint_switch   party slot to send in after a faint
 *   release        party full after a catch: 0-5 = release that slot to keep obs.caught, 6 = let it go
 * Remaining inputs (biome choice, prompts) are handled by scripted defaults in
 * `watchdogTick` and counted in `info.autoInputs`.
 *
 * Only runs when RL_PORT is set, so normal `pnpm test` runs skip it.
 *
 * Source of truth is sim/ in the Pokerogue RL repo; pokerogue/test/rl/ holds a copy made by
 * `python -m pokerogue_rl.install_sim` (Vite can't load test files from outside pokerogue/).
 * Start: RL_PORT=5555 pnpm exec vitest run test/rl/sim-server.test.ts
 */

import { BattleScene } from "#app/battle-scene";
import { getGameMode } from "#app/game-mode";
import type { Phase } from "#app/phase";
import { allMoves } from "#data/data-lists";
import { getCriticalCaptureChance, getPokeballCatchMultiplier } from "#data/pokeball";
import { getStatusEffectCatchRateMultiplier } from "#data/status-effect";
import { AbilityId } from "#enums/ability-id";
import { BattleStyle } from "#enums/battle-style";
import { BattlerIndex } from "#enums/battler-index";
import { BattleType } from "#enums/battle-type";
import { BiomeId } from "#enums/biome-id";
import { Button } from "#enums/buttons";
import { Command } from "#enums/command";
import { GameModes } from "#enums/game-modes";
import { MoveId } from "#enums/move-id";
import { Nature } from "#enums/nature";
import { PokeballType } from "#enums/pokeball";
import { SpeciesId } from "#enums/species-id";
import { TrainerType } from "#enums/trainer-type";
import { UiMode } from "#enums/ui-mode";
import type { EnemyPokemon, Pokemon } from "#field/pokemon";
import { HealShopCostModifier } from "#modifiers/modifier";
import {
  FusePokemonModifierType,
  getPartyLuckValue,
  getPlayerShopModifierTypeOptionsForWave,
  type ModifierType,
  type ModifierTypeOption,
  PokemonModifierType,
  PokemonMoveModifierType,
  RememberMoveModifierType,
  TmModifierType,
} from "#modifiers/modifier-type";
import type { Move } from "#moves/move";
import { getMoveTargets } from "#moves/move-utils";
import type { AttemptCapturePhase } from "#phases/attempt-capture-phase";
import type { CommandPhase } from "#phases/command-phase";
import type { LearnMovePhase } from "#phases/learn-move-phase";
import type { PokemonPhase } from "#phases/pokemon-phase";
import type { SelectModifierPhase } from "#phases/select-modifier-phase";
import { SelectStarterPhase } from "#phases/select-starter-phase";
import { GameManager } from "#test/framework/game-manager";
import { PromptHandler } from "#test/helpers/prompt-handler";
import { generateStarters } from "#test/utils/game-manager-utils";
import type { StarterMoveset } from "#types/save-data";
import type { StarterSpeciesId } from "#types/starter-species-id";
import { PartyOption } from "#ui/party-ui-handler";
import { getStarterMoves } from "#ui/utils/starter-select-ui-utils";
import { NumberHolder, randomString } from "#utils/common";
import net from "node:net";
import Phaser from "phaser";
import { it, vi } from "vitest";

const PORT = Number(process.env.RL_PORT ?? 0);
/** Command actions: 0-3 moves, 4-9 switch to party slot, 10-14 throw ball type (Poke, Great, Ultra, Rogue, Master). */
const NUM_COMMAND_ACTIONS = 15;
const NUM_BALL_TYPES = 5;
/** Release decision: 0-5 release that party slot to keep the catch, 6 lets the catch go. */
const RELEASE_DECLINE = 6;
const MAX_PHASES_PER_STEP = 5000;
/** Display-only phases that crash against the mocked UI; skipping them has no gameplay effect. */
const SKIPPED_PHASES = new Set(["ScanIvsPhase"]);
/**
 * Ends the episode if the agent makes this many decisions without the wave advancing.
 * Some decisions can loop by design (declining a TM returns to the reward screen).
 */
const MAX_DECISIONS_PER_WAVE = 1000;
const STALL_TIMEOUT_MS = 10_000;

// GameManager forces max rolls on battle RNG for deterministic tests; keep the real one for RL.
const realRandBattleSeedInt = BattleScene.prototype.randBattleSeedInt;

function log(msg: string): void {
  process.stderr.write(`[sim] ${msg}\n`);
}

/** Item name like "POTION". Shop items are built without an `id`, so fall back to the locale key's tail. */
function itemId(type: ModifierType): string {
  return type.id || type.localeKey.split(".").pop()!;
}

interface ResetOpts {
  seed?: string;
  mode?: "endless" | "classic";
  starters?: string[];
  maxWave?: number;
  /** IV applied to every stat of every starter; default 31. */
  ivs?: number;
  nature?: Nature;
  /** Indices of starters that start with Pokerus (1.5x EXP, spreads to party neighbours); default [0]. */
  pokerus?: number[];
}

interface RewardOption {
  kind: "reward" | "shop" | "reroll" | "skip";
  type?: ModifierType;
  /** -1 for free rewards. */
  cost: number;
  legal: boolean;
}

type Decision =
  | { kind: "command"; phase: CommandPhase }
  | { kind: "reward"; phase: SelectModifierPhase; options: RewardOption[] }
  | { kind: "reward_target"; phase: SelectModifierPhase; option: RewardOption }
  | { kind: "learn_move"; phase: LearnMovePhase }
  | { kind: "target"; phase: PokemonPhase }
  // Mid-phase decisions: the phase is still running while it waits on these.
  | { kind: "faint_switch"; phase: Phase }
  | { kind: "release"; phase: AttemptCapturePhase };

class Sim {
  private game!: GameManager;
  private watchdog: NodeJS.Timeout | undefined;
  private maxWave = Number.POSITIVE_INFINITY;
  private autoInputs: Record<string, number> = {};
  private lastStall = "";
  private stallTicks = 0;
  /** The decision the agent must make next; while set, the watchdog stays hands-off. */
  private decision: Decision | undefined;
  /** A LearnMovePhase the agent has already decided on, so advance() runs it instead of asking again. */
  private approvedLearnMove: LearnMovePhase | undefined;
  private decisionWave = 0;
  private decisionsThisWave = 0;
  /** The scene's MockClock, patched by installManualClock. */
  private clock: any;

  constructor(private readonly phaserGame: Phaser.Game) {}

  private get scene(): BattleScene {
    return this.game.scene;
  }

  private get interceptorState(): string {
    return this.game.phaseInterceptor["state"];
  }

  async reset(opts: ResetOpts) {
    this.stopWatchdog();
    this.decision = undefined;
    this.approvedLearnMove = undefined;
    this.decisionWave = 0;
    this.decisionsThisWave = 0;
    clearInterval(PromptHandler.runInterval);
    PromptHandler.runInterval = undefined;
    // PromptHandler spies on ui.setModeInternal; re-spying an existing spy would recurse forever.
    vi.restoreAllMocks();

    this.game = new GameManager(this.phaserGame);
    BattleScene.prototype.randBattleSeedInt = realRandBattleSeedInt;
    this.installCaptureCounter();
    this.installManualClock();
    this.manualClock = false; // the test framework's helpers below need the wall-clock interval
    this.game.settings.battleStyle(BattleStyle.SET);
    this.maxWave = opts.maxWave ?? Number.POSITIVE_INFINITY;
    this.autoInputs = {};

    const mode = opts.mode === "classic" ? GameModes.CLASSIC : GameModes.ENDLESS;
    const species = (opts.starters ?? ["CHARMANDER", "SQUIRTLE", "BULBASAUR"]).map(name => {
      const id = SpeciesId[name as keyof typeof SpeciesId];
      if (id === undefined) {
        throw new Error(`Unknown species: ${name}`);
      }
      return id;
    });

    await this.game.runToTitle();
    this.game.onNextPrompt("TitlePhase", UiMode.TITLE, () => {
      this.scene.gameMode = getGameMode(mode);
      const starters = generateStarters(this.scene, species);
      // The test helper rolls these randomly before the run seed exists; pin them for reproducibility.
      const pokerus = opts.pokerus ?? [0];
      for (const [i, starter] of starters.entries()) {
        Object.assign(starter, {
          pokerus: pokerus.includes(i),
          ivs: new Array(6).fill(opts.ivs ?? 31),
          nature: opts.nature ?? Nature.HARDY,
          abilityIndex: 0,
          shiny: false,
          variant: 0,
          female: false,
        });
        // Outside daily runs, movesets come from the starter-select screen; use its fresh-save default.
        if (!starter.moveset?.length) {
          starter.moveset = getStarterMoves(starter.speciesId as StarterSpeciesId, starter.formIndex).slice(0, 4) as StarterMoveset;
        }
      }
      // generateStarters() forces seed "test", so seed afterwards.
      this.scene.setSeed(opts.seed ?? randomString(24));
      this.scene.resetSeed();
      this.scene.phaseManager.pushNew("EncounterPhase", false);
      new SelectStarterPhase().initBattle(starters);
    });
    await this.game.phaseInterceptor.to("EncounterPhase");

    this.manualClock = true;
    this.startWatchdog();
    return this.advance();
  }

  /**
   * The test harness's MockClock advances on a real 1ms setInterval, so game timers (delayedCall,
   * text delays) fire by wall time and their order relative to other game logic varies under CPU
   * load, making runs nondeterministic. While `manualClock` is set, ignore the interval and advance
   * the clock one fixed step per poll in `waitUntil` instead.
   */
  private installManualClock() {
    const clock = this.scene.time as any;
    if (!clock.rlTick) {
      // The scene, and its clock, persist across resets: patch once.
      const preUpdate = clock.preUpdate.bind(clock);
      const update = clock.update.bind(clock);
      const tick = () => {
        preUpdate();
        update(clock.systems.game.loop.time, 1);
      };
      clock.preUpdate = () => {};
      clock.update = () => {
        if (!clock.rlManual) {
          tick();
        }
      };
      clock.rlTick = tick;
    }
    this.clock = clock;
  }

  private set manualClock(on: boolean) {
    this.clock.rlManual = on;
  }

  /**
   * The test harness mocks tween counters to fire only `onComplete`. Ball shake checks run in
   * `onRepeat`, so every throw would succeed. Run the capture shake counter faithfully instead.
   * Other counters (battle animations, particles) are purely visual and keep the cheap mock.
   */
  private installCaptureCounter() {
    const tweens = this.scene.tweens as any;
    if (tweens.rlCaptureCounter) {
      return; // the scene, and its tween mock, persist across resets
    }
    const mockAddCounter = tweens.addCounter;
    tweens.addCounter = (config: any) => {
      const isShake =
        this.scene.phaseManager.getCurrentPhase()?.is("AttemptCapturePhase")
        && config.yoyo
        && config.onRepeat
        && config.repeat > 0;
      if (!isShake) {
        return mockAddCounter(config);
      }
      let stopped = false;
      const counter = {
        stop: () => {
          stopped = true;
        },
        getValue: () => 0,
      };
      // Callbacks reference the counter through the caller's variable, so run after returning it.
      queueMicrotask(() => {
        for (let i = 0; i < config.repeat && !stopped; i++) {
          config.onRepeat(counter);
        }
        if (!stopped) {
          config.onComplete?.(counter);
        }
      });
      return counter;
    };
    tweens.rlCaptureCounter = true;
  }

  async step(action: number) {
    const decision = this.decision;
    if (!decision) {
      throw new Error("step() called with no pending decision");
    }
    const legal = this.legalMask(decision);
    const invalid = !legal[action];
    if (invalid) {
      // Fall back to the first legal action so the episode can continue.
      action = legal.findIndex(Boolean);
      if (action < 0) {
        throw new Error(`No legal action for decision ${decision.kind}`);
      }
    }
    this.decision = undefined;

    const result = await this.act(decision, action);
    result.info.invalidAction = invalid;
    return result;
  }

  private async act(decision: Decision, action: number) {
    switch (decision.kind) {
      case "command": {
        const phase = decision.phase;
        const ok =
          action < 4
            ? phase.handleCommand(Command.FIGHT, action)
            : action < 10
              ? phase.handleCommand(Command.POKEMON, action - 4, false)
              : phase.handleCommand(Command.BALL, action - 10);
        if (!ok) {
          throw new Error(`CommandPhase rejected legal action ${action}`);
        }
        return this.advance();
      }

      case "reward": {
        const option = decision.options[action];
        const phase = decision.phase;
        if (option.kind === "skip") {
          this.scene.ui.setMode(UiMode.MESSAGE);
          phase.end();
          return this.advance();
        }
        if (option.kind === "reroll") {
          phase["rerollModifiers"]();
          return this.advance();
        }
        if (option.type instanceof PokemonModifierType) {
          return this.decide({ kind: "reward_target", phase, option });
        }
        return this.applyReward(phase, option);
      }

      case "reward_target":
        return this.applyReward(decision.phase, decision.option, action);

      case "target":
        this.setTarget(decision.phase, action);
        this.scene.phaseManager.shiftPhase(); // the SelectTargetPhase was never started
        return this.advance();

      case "learn_move": {
        const phase = decision.phase as any;
        // Answer the "replace a move?" prompt ourselves instead of through the UI.
        phase.replaceMoveCheck =
          action < 4 ? (move: Move, pokemon: Pokemon) => phase.learnMove(action, move, pokemon) : () => phase.end();
        this.approvedLearnMove = decision.phase;
        return this.advance();
      }

      case "faint_switch": {
        const handler = this.scene.ui.getHandler() as any;
        handler.setCursor(action);
        handler.processInput(Button.ACTION); // select slot
        handler.processInput(Button.ACTION); // "Send out"
        return this.resume(decision.phase, UiMode.PARTY);
      }

      case "release": {
        const ui = this.scene.ui;
        // Party-full menu: [Summary, Pokedex, Yes (keep it, release someone), No (let it go)].
        const options = (ui.getHandler() as any).config.options;
        if (action === RELEASE_DECLINE) {
          options[3].handler();
          return this.resume(decision.phase, UiMode.OPTION_SELECT);
        }
        options[2].handler();
        await this.waitUntil(
          () => ui.mode === UiMode.PARTY && !!(ui.getHandler() as any).selectCallback,
          "release menu",
        );
        // Do what PartyUiHandler.doRelease does, minus its message and sprite refresh.
        const partyHandler = ui.getHandler() as any;
        this.scene.removePartyMemberModifiers(action);
        this.scene.getPlayerParty().splice(action, 1)[0].destroy();
        const callback = partyHandler.selectCallback;
        partyHandler.selectCallback = null;
        callback(action, PartyOption.RELEASE); // adds the caught Pokemon at this slot
        return this.resume(decision.phase, UiMode.PARTY);
      }
    }
  }

  /** Continue a phase that paused for a mid-phase decision, once its UI has moved on from `fromMode`. */
  private async resume(phase: Phase, fromMode: UiMode) {
    await this.waitUntil(() => this.scene.ui.mode !== fromMode || this.interceptorState !== "running", "UI to move on");
    const paused = await this.waitForPhase(phase);
    return paused ? this.decide(paused) : this.advance();
  }

  private applyReward(phase: SelectModifierPhase, option: RewardOption, slot?: number) {
    const type = option.type!;
    let modifier;
    if (type instanceof PokemonModifierType) {
      const pokemon = this.scene.getPlayerParty()[slot!];
      modifier =
        type instanceof PokemonMoveModifierType
          ? type.newModifier(pokemon, this.pickMoveForItem(pokemon, type))
          : type.newModifier(pokemon);
    } else {
      modifier = type.newModifier();
    }
    if (!modifier) {
      throw new Error(`Could not create modifier for ${type.id}`);
    }
    phase["applyModifier"](modifier, option.cost, false);
    // Shop purchases leave the reward screen open; free rewards end it.
    if (option.kind === "shop") {
      return this.decide({ kind: "reward", phase, options: this.rewardOptions(phase) });
    }
    return this.advance();
  }

  /** Run phases until the agent must decide something or the run ends. */
  private async advance() {
    const pm = this.scene.phaseManager;
    for (let i = 0; i < MAX_PHASES_PER_STEP; i++) {
      await this.waitUntil(() => this.interceptorState !== "interrupted", "phase awaiting input");
      const phase = pm.getCurrentPhase();
      const name = phase.phaseName;

      if (name === "GameOverPhase") {
        return this.result(true, "game_over");
      }
      if (this.scene.currentBattle && this.scene.currentBattle.waveIndex > this.maxWave) {
        return this.result(true, "max_wave");
      }
      if (SKIPPED_PHASES.has(name)) {
        pm.shiftPhase();
        continue;
      }
      if (name === "SelectTargetPhase") {
        // Resolved without its UI, which otherwise needs timing-dependent input.
        const targetPhase = phase as PokemonPhase;
        const mask = this.battleTargetMask(targetPhase);
        if (mask.filter(Boolean).length > 1) {
          return this.decide({ kind: "target", phase: targetPhase });
        }
        // Multi-target moves already have targets; a single legal target needs no decision.
        this.setTarget(targetPhase, mask.indexOf(true));
        pm.shiftPhase();
        continue;
      }
      if (name === "LearnMovePhase" && phase !== this.approvedLearnMove && this.needsLearnDecision(phase as LearnMovePhase)) {
        return this.decide({ kind: "learn_move", phase: phase as LearnMovePhase });
      }

      const paused = await this.runPhase(phase);
      if (paused) {
        return this.decide(paused);
      }

      if (this.interceptorState !== "interrupted" || pm.getCurrentPhase() !== phase) {
        continue;
      }
      if (name === "CommandPhase" && this.scene.ui.mode === UiMode.COMMAND) {
        return this.decide({ kind: "command", phase: phase as CommandPhase });
      }
      if (name === "SelectModifierPhase" && this.scene.ui.mode === UiMode.MODIFIER_SELECT) {
        const rewardPhase = phase as SelectModifierPhase;
        return this.decide({ kind: "reward", phase: rewardPhase, options: this.rewardOptions(rewardPhase) });
      }
    }
    throw new Error(`advance(): exceeded ${MAX_PHASES_PER_STEP} phases without a decision`);
  }

  /**
   * Which battler indices (player 0/1, enemy 2/3) a pending single-target move may be aimed at.
   * All false when the move already has its targets (multi-target moves).
   */
  private battleTargetMask(phase: PokemonPhase): boolean[] {
    const mask = [false, false, false, false];
    const command = this.scene.currentBattle.turnCommands[phase.fieldIndex];
    if (!command?.move || (command.move.targets?.length ?? 0) > 0) {
      return mask;
    }
    const user = this.scene.getPlayerField()[phase.fieldIndex];
    const field = this.scene.getField();
    const { targets } = getMoveTargets(user, command.move.move);
    for (const t of targets) {
      // Mirrors SelectTargetPhase, which rejects targets with a restricting tag (Heal Block + Pollen Puff).
      mask[t] = !user.getTargetRestrictingTag(command.move.move, field[t]);
    }
    if (!mask.some(Boolean)) {
      for (const t of targets) {
        mask[t] = true;
      }
    }
    return mask;
  }

  private setTarget(phase: PokemonPhase, target: number) {
    const command = this.scene.currentBattle.turnCommands[phase.fieldIndex];
    if (command && target >= 0) {
      command.targets = [target as BattlerIndex];
    }
  }

  private decide(decision: Decision) {
    const wave = this.scene.currentBattle?.waveIndex ?? 0;
    if (wave !== this.decisionWave) {
      this.decisionWave = wave;
      this.decisionsThisWave = 0;
    }
    if (++this.decisionsThisWave > MAX_DECISIONS_PER_WAVE) {
      this.decision = undefined;
      return this.result(true, "decision_limit");
    }
    this.decision = decision;
    return this.result(false);
  }

  /**
   * Start a phase and wait until it ends, interrupts for UI input, or pauses on a mid-phase
   * agent decision (returned). Polls every tick rather than the test framework's 50ms.
   */
  private async runPhase(phase: Phase): Promise<Decision | undefined> {
    const interceptor = this.game.phaseInterceptor;
    interceptor["target"] = phase.phaseName;
    interceptor["state"] = "running";
    phase.start();
    return this.waitForPhase(phase);
  }

  private async waitForPhase(phase: Phase): Promise<Decision | undefined> {
    let paused: Decision | undefined;
    await this.waitUntil(() => {
      paused = this.interceptorState === "running" ? this.midPhaseDecision(phase) : undefined;
      return paused !== undefined || this.interceptorState !== "running";
    }, `running ${phase.phaseName}`);
    return paused;
  }

  private midPhaseDecision(phase: Phase): Decision | undefined {
    const ui = this.scene.ui;
    if (!(ui.getHandler() as any)?.active) {
      return undefined;
    }
    if (phase.is("SwitchPhase") && ui.mode === UiMode.PARTY) {
      return { kind: "faint_switch", phase };
    }
    if (phase.is("AttemptCapturePhase") && ui.mode === UiMode.OPTION_SELECT) {
      return { kind: "release", phase: phase as AttemptCapturePhase };
    }
    return undefined;
  }

  private async waitUntil(cond: () => boolean, what: string) {
    const start = Date.now();
    while (!cond()) {
      if (Date.now() - start > STALL_TIMEOUT_MS) {
        const phase = this.scene.phaseManager.getCurrentPhase()?.phaseName;
        throw new Error(`Stalled ${STALL_TIMEOUT_MS}ms while ${what} (phase=${phase}, ui=${UiMode[this.scene.ui.mode]})`);
      }
      if (this.clock.rlManual) {
        this.clock.rlTick();
      }
      await new Promise(resolve => setImmediate(resolve));
    }
  }

  // #region Decision rules

  private legalMask(decision: Decision): boolean[] {
    switch (decision.kind) {
      case "command":
        return this.commandMask(decision.phase);
      case "reward":
        return decision.options.map(o => o.legal);
      case "reward_target":
        return this.targetMask(decision.option.type as PokemonModifierType);
      case "learn_move":
        return [true, true, true, true, true];
      case "target":
        return this.battleTargetMask(decision.phase);
      case "faint_switch": {
        const party = this.scene.getPlayerParty();
        const battlers = this.scene.currentBattle.getBattlerCount();
        return [0, 1, 2, 3, 4, 5].map(
          i => i >= battlers && i < party.length && party[i].isAllowedInBattle() && !party[i].isOnField(),
        );
      }
      case "release": {
        // Any party slot can be released; the battle ends after a catch.
        const size = this.scene.getPlayerParty().length;
        return [...[0, 1, 2, 3, 4, 5].map(i => i < size), true];
      }
    }
  }

  private commandMask(phase: CommandPhase): boolean[] {
    const mask = new Array<boolean>(NUM_COMMAND_ACTIONS).fill(false);
    const pokemon = phase.getPokemon();
    const moveset = pokemon.getMoveset();
    const usable = moveset.map(m => m.isUsable(pokemon, false, true)[0]);
    if (usable.some(Boolean)) {
      usable.forEach((u, i) => {
        mask[i] = u;
      });
    } else if (moveset.length > 0) {
      mask[0] = true; // resolves to Struggle
    }
    if (!pokemon.isTrapped()) {
      const party = this.scene.getPlayerParty();
      const battlers = this.scene.currentBattle.getBattlerCount();
      for (let slot = battlers; slot < Math.min(party.length, 6); slot++) {
        mask[4 + slot] = party[slot].isAllowedInBattle() && !party[slot].isOnField();
      }
    }
    const target = this.ballTarget();
    for (let ball = 0; ball < NUM_BALL_TYPES; ball++) {
      mask[10 + ball] = target !== undefined && this.canThrowBall(target, ball);
    }
    return mask;
  }

  /**
   * The wild Pokemon a ball would target, if throwing is allowed at all. Mirrors
   * CommandPhase.checkCanUseBall/handleBallCommand, whose failure paths leave UI text behind.
   */
  private ballTarget(): EnemyPokemon | undefined {
    const { currentBattle: battle, arena } = this.scene;
    if (
      battle.battleType !== BattleType.WILD
      || battle.isBattleMysteryEncounter()
      // Conservative: End-biome catch rules depend on mode and boss status; just disallow there.
      || arena.biomeId === BiomeId.END
    ) {
      return undefined;
    }
    const targets = this.scene.getEnemyField().filter(p => p.isActive(true));
    return targets.length === 1 ? (targets[0] as EnemyPokemon) : undefined;
  }

  private canThrowBall(target: EnemyPokemon, ball: PokeballType): boolean {
    if ((this.scene.pokeballCounts[ball] ?? 0) <= 0) {
      return false;
    }
    // Bosses with a broken shield can only be caught with a Master Ball.
    const shielded =
      target.isBoss() && target.bossSegmentIndex >= 1 && !target.hasAbility(AbilityId.WONDER_GUARD, false, true);
    return !shielded || ball === PokeballType.MASTER_BALL;
  }

  /** Probability a ball catches the target, per AttemptCapturePhase (ignoring the shiny bonus). */
  private catchChance(target: EnemyPokemon, ball: PokeballType): number {
    if (!this.canThrowBall(target, ball)) {
      return 0;
    }
    const multiplier = getPokeballCatchMultiplier(ball);
    if (multiplier === -1) {
      return 1;
    }
    const maxHp3 = 3 * target.getMaxHp();
    const statusMultiplier = target.status ? getStatusEffectCatchRateMultiplier(target.status.effect) : 1;
    const rate = Math.round(
      (((maxHp3 - 2 * target.hp) * target.species.catchRate * multiplier) / maxHp3) * statusMultiplier,
    );
    if (rate >= 255) {
      return 1;
    }
    const shake = Math.round(65536 / Math.pow(255 / rate, 0.1875)) / 65536;
    const critical = getCriticalCaptureChance(rate) / 256;
    // Critical captures need one shake check, normal ones three.
    return critical * shake + (1 - critical) * shake ** 3;
  }

  private rewardOptions(phase: SelectModifierPhase): RewardOption[] {
    const scene = this.scene;
    const options: RewardOption[] = (phase["typeOptions"] as ModifierTypeOption[]).map(o => ({
      kind: "reward",
      type: o.type,
      cost: -1,
      legal: this.canApply(o.type),
    }));
    if (scene.gameMode.getShopStatus()) {
      const shop = getPlayerShopModifierTypeOptionsForWave(scene.currentBattle.waveIndex, scene.getWaveMoneyAmount(1));
      for (const o of shop) {
        const cost = new NumberHolder(o.cost);
        scene.applyModifier(HealShopCostModifier, true, cost);
        options.push({ kind: "shop", type: o.type, cost: cost.value, legal: scene.money >= cost.value && this.canApply(o.type) });
      }
    }
    const rerollCost = phase.getRerollCost(scene.lockModifierTiers);
    options.push({ kind: "reroll", cost: rerollCost, legal: rerollCost >= 0 && scene.money >= rerollCost });
    options.push({ kind: "skip", cost: 0, legal: true });
    return options;
  }

  private canApply(type: ModifierType): boolean {
    // Fusion and Memory Mushroom need multi-step menus that aren't modelled yet.
    if (type instanceof FusePokemonModifierType || type instanceof RememberMoveModifierType) {
      return false;
    }
    if (type instanceof PokemonModifierType) {
      return this.targetMask(type).some(Boolean);
    }
    return true;
  }

  private targetMask(type: PokemonModifierType): boolean[] {
    return this.scene.getPlayerParty().map(p => {
      if (type.selectFilter && type.selectFilter(p) !== null) {
        return false;
      }
      return !(type instanceof PokemonMoveModifierType) || this.pickMoveForItem(p, type) >= 0;
    });
  }

  /** Move slot an Ether/PP Up-style item applies to: the eligible move with the least PP left. */
  private pickMoveForItem(pokemon: Pokemon, type: PokemonMoveModifierType): number {
    let best = -1;
    let bestRatio = Number.POSITIVE_INFINITY;
    pokemon.getMoveset().forEach((m, i) => {
      if (type.moveSelectFilter && type.moveSelectFilter(m) !== null) {
        return;
      }
      const ratio = 1 - m.ppUsed / m.getMovePp();
      if (ratio < bestRatio) {
        best = i;
        bestRatio = ratio;
      }
    });
    return best;
  }

  private needsLearnDecision(phase: LearnMovePhase): boolean {
    const move = allMoves[phase["moveId"] as MoveId];
    const moveset = phase.getPokemon().getMoveset();
    return !move.isUnimplemented && moveset.length >= 4 && !moveset.some(m => m.moveId === move.id);
  }

  // #endregion

  // #region Scripted defaults for non-agent inputs

  private startWatchdog() {
    this.watchdog = setInterval(() => this.watchdogTick(), 1);
  }

  private stopWatchdog() {
    clearInterval(this.watchdog);
    this.watchdog = undefined;
  }

  private watchdogTick() {
    const ui = this.scene.ui;
    const handler = ui.getHandler() as any;
    const phaseName = this.scene.phaseManager.getCurrentPhase()?.phaseName;
    if (this.decision || !handler?.active || this.interceptorState === "idling") {
      return;
    }
    // Only act once the same (phase, mode) has persisted for a tick, so transitions settle first.
    const key = `${phaseName}:${ui.mode}`;
    if (key !== this.lastStall) {
      this.lastStall = key;
      this.stallTicks = 0;
      return;
    }
    if (++this.stallTicks < 2) {
      return;
    }
    this.stallTicks = 0;

    // Agent decisions; never auto-answer.
    const agentScreen =
      ui.mode === UiMode.COMMAND
      || ui.mode === UiMode.FIGHT
      || ui.mode === UiMode.MODIFIER_SELECT
      || (ui.mode === UiMode.PARTY && (phaseName === "SwitchPhase" || phaseName === "AttemptCapturePhase"))
      || (ui.mode === UiMode.OPTION_SELECT && phaseName === "AttemptCapturePhase");
    if (agentScreen) {
      return;
    }

    switch (ui.mode) {
      case UiMode.PARTY: {
        handler.processInput(Button.CANCEL);
        this.count(`party_cancel:${phaseName}`);
        return;
      }
      case UiMode.CONFIRM:
        handler.processInput(Button.CANCEL);
        this.count(`confirm:${phaseName}`);
        return;
      case UiMode.OPTION_SELECT:
      case UiMode.TARGET_SELECT:
        handler.processInput(Button.ACTION);
        this.count(`${UiMode[ui.mode]}:${phaseName}`);
        return;
      default:
        if (handler.awaitingActionInput) {
          handler.processInput(Button.ACTION);
          this.count(`${UiMode[ui.mode]}:${phaseName}`);
        }
    }
  }

  private count(key: string) {
    this.autoInputs[key] = (this.autoInputs[key] ?? 0) + 1;
  }

  // #endregion

  // #region Observation

  private encodeMove(moveId: MoveId) {
    const move = allMoves[moveId];
    return {
      id: moveId,
      type: move.type,
      category: move.category,
      power: move.power,
      accuracy: move.accuracy,
      priority: move.priority,
    };
  }

  private encodePokemon(p: Pokemon, withDamage: boolean) {
    const enemies = withDamage ? this.scene.getEnemyField() : [];
    return {
      species: p.species.speciesId,
      bst: p.species.baseTotal,
      catchRate: p.species.catchRate,
      shiny: p.isShiny(),
      luck: p.getLuck(),
      pokerus: p.pokerus,
      level: p.level,
      hp: p.hp,
      maxHp: p.getMaxHp(),
      stats: [0, 1, 2, 3, 4, 5].map(s => p.getStat(s)),
      statStages: p.isOnField() ? p.getStatStages() : [],
      status: p.status?.effect ?? 0,
      types: p.getTypes(),
      ability: p.getAbility().id,
      fainted: p.isFainted(),
      onField: p.isOnField(),
      boss: p.isBoss(),
      moves: p.getMoveset().map(m => {
        const move = m.getMove();
        return {
          ...this.encodeMove(m.moveId),
          pp: m.getMovePp() - m.ppUsed,
          maxPp: m.getMovePp(),
          // Per enemy on field: type multiplier and simulated (max-roll, no RNG) damage / current HP.
          eff: enemies.map(e => e.getAttackTypeEffectiveness(move.type, { source: p, move })),
          dmg: enemies.map(e => this.damageFraction(p, e, move)),
        };
      }),
    };
  }

  private damageFraction(source: Pokemon, target: Pokemon, move: Move): number | null {
    if (target.hp <= 0) {
      return null;
    }
    try {
      return target.getAttackDamage({ source, move, simulated: true }).damage / target.hp;
    } catch {
      return null;
    }
  }

  private result(done: boolean, reason?: string) {
    const scene = this.scene;
    const battle = scene.currentBattle;
    const d = done ? undefined : this.decision;
    const withDamage = d?.kind === "command" || d?.kind === "faint_switch" || d?.kind === "target";
    const obs: Record<string, unknown> = {
      kind: d?.kind ?? "done",
      legal: d ? this.legalMask(d) : [],
      wave: battle?.waveIndex ?? 0,
      turn: battle?.turn ?? 0,
      double: battle?.double ?? false,
      battleType: battle ? BattleType[battle.battleType] : null,
      trainer: battle?.trainer ? TrainerType[battle.trainer.config.trainerType] : null,
      money: scene.money,
      biome: scene.arena?.biomeId,
      balls: [0, 1, 2, 3, 4].map(b => scene.pokeballCounts[b as PokeballType] ?? 0),
      luck: getPartyLuckValue(scene.getPlayerParty()),
      party: scene.getPlayerParty().map(p => this.encodePokemon(p, withDamage)),
      enemies: scene.getEnemyField().map(p => this.encodePokemon(p, false)),
    };
    if (d?.kind === "command") {
      obs.actor = d.phase.getFieldIndex();
      const target = this.ballTarget();
      obs.catchChance = [0, 1, 2, 3, 4].map(b => (target ? this.catchChance(target, b) : 0));
    } else if (d?.kind === "reward") {
      obs.options = d.options.map(o => ({
        kind: o.kind,
        id: o.type ? itemId(o.type) : o.kind.toUpperCase(),
        tier: o.type?.tier ?? null,
        cost: o.cost,
        needsTarget: o.type instanceof PokemonModifierType,
        tmMove: o.type instanceof TmModifierType ? this.encodeMove(o.type.moveId) : null,
        legal: o.legal,
      }));
    } else if (d?.kind === "reward_target") {
      obs.item = { id: itemId(d.option.type!), cost: d.option.cost };
    } else if (d?.kind === "learn_move") {
      obs.learn = { slot: d.phase["partyMemberIndex"], move: this.encodeMove(d.phase["moveId"]) };
    } else if (d?.kind === "target") {
      const command = scene.currentBattle.turnCommands[d.phase.fieldIndex]!;
      obs.actor = d.phase.fieldIndex;
      obs.move = { slot: command.cursor, ...this.encodeMove(command.move!.move) };
    } else if (d?.kind === "release") {
      obs.caught = this.encodePokemon(d.phase.getPokemon(), false);
    }
    return {
      obs,
      done,
      info: { reason, autoInputs: { ...this.autoInputs } } as Record<string, unknown>,
    };
  }

  // #endregion
}

function serve(sim: Sim): Promise<void> {
  return new Promise((resolve, reject) => {
    const server = net.createServer(socket => {
      let buffer = "";
      let queue = Promise.resolve();
      socket.setNoDelay(true);
      socket.on("data", chunk => {
        buffer += chunk.toString("utf8");
        let nl: number;
        while ((nl = buffer.indexOf("\n")) >= 0) {
          const line = buffer.slice(0, nl);
          buffer = buffer.slice(nl + 1);
          queue = queue.then(() => handle(line));
        }
      });

      const send = (msg: object) => socket.write(`${JSON.stringify(msg)}\n`);
      async function handle(line: string) {
        let req: any;
        try {
          req = JSON.parse(line);
          switch (req.cmd) {
            case "reset":
              send({ ok: true, ...(await sim.reset(req)) });
              break;
            case "step":
              send({ ok: true, ...(await sim.step(Number(req.action))) });
              break;
            case "close":
              send({ ok: true });
              socket.end();
              break;
            case "shutdown":
              send({ ok: true });
              socket.end();
              server.close();
              resolve();
              break;
            default:
              send({ ok: false, error: `unknown cmd ${req.cmd}` });
          }
        } catch (err) {
          log(`error handling ${req?.cmd}: ${(err as Error).stack ?? err}`);
          send({ ok: false, error: String((err as Error).message ?? err) });
        }
      }
    });
    server.on("error", reject);
    server.listen(PORT, "127.0.0.1", () => {
      log(`listening on 127.0.0.1:${(server.address() as net.AddressInfo).port}`);
    });
  });
}

it.skipIf(!process.env.RL_PORT)(
  "RL simulator server",
  async () => {
    const phaserGame = new Phaser.Game({ type: Phaser.HEADLESS });
    await serve(new Sim(phaserGame));
  },
  2 ** 31 - 1,
);
