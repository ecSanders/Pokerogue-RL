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
 * Actions at a CommandPhase: 0-3 = use move slot, 4+k = switch to party slot k.
 * Every other input (rewards, faint switches, learn-move, biome choice, prompts) is handled
 * by scripted defaults in `watchdogTick` and counted in `info.autoInputs`.
 *
 * Only runs when RL_PORT is set, so normal `pnpm test` runs skip it.
 *
 * Source of truth is sim/ in the Pokerogue RL repo; pokerogue/test/rl/ holds a copy made by
 * `python -m pokerogue_rl.install_sim` (Vite can't load test files from outside pokerogue/).
 * Start: RL_PORT=5555 pnpm exec vitest run test/rl/sim-server.test.ts
 */

import { BattleScene } from "#app/battle-scene";
import { getGameMode } from "#app/game-mode";
import { BattleStyle } from "#enums/battle-style";
import { Button } from "#enums/buttons";
import { Command } from "#enums/command";
import { GameModes } from "#enums/game-modes";
import { Nature } from "#enums/nature";
import { SpeciesId } from "#enums/species-id";
import { UiMode } from "#enums/ui-mode";
import type { Pokemon } from "#field/pokemon";
import type { CommandPhase } from "#phases/command-phase";
import { SelectStarterPhase } from "#phases/select-starter-phase";
import { GameManager } from "#test/framework/game-manager";
import { PromptHandler } from "#test/helpers/prompt-handler";
import { generateStarters } from "#test/utils/game-manager-utils";
import type { StarterMoveset } from "#types/save-data";
import { getStarterMoves } from "#ui/utils/starter-select-ui-utils";
import { randomString } from "#utils/common";
import net from "node:net";
import Phaser from "phaser";
import { it, vi } from "vitest";

const PORT = Number(process.env.RL_PORT ?? 0);
const NUM_ACTIONS = 10;
/** Phases skipped outright instead of being run (their decisions aren't modelled yet). */
const SKIPPED_PHASES = new Set(["LearnMovePhase"]);
const MAX_PHASES_PER_STEP = 5000;
const STALL_TIMEOUT_MS = 10_000;

// GameManager forces max rolls on battle RNG for deterministic tests; keep the real one for RL.
const realRandBattleSeedInt = BattleScene.prototype.randBattleSeedInt;

function log(msg: string): void {
  process.stderr.write(`[sim] ${msg}\n`);
}

interface ResetOpts {
  seed?: string;
  mode?: "endless" | "classic";
  starters?: string[];
  maxWave?: number;
  /** IV applied to every stat of every starter; default 31. */
  ivs?: number;
  nature?: Nature;
}

class Sim {
  private game!: GameManager;
  private watchdog?: NodeJS.Timeout;
  private maxWave = Number.POSITIVE_INFINITY;
  private autoInputs: Record<string, number> = {};
  private lastStall = "";
  private stallTicks = 0;

  constructor(private readonly phaserGame: Phaser.Game) {}

  private get scene(): BattleScene {
    return this.game.scene;
  }

  private get interceptorState(): string {
    return this.game.phaseInterceptor["state"];
  }

  async reset(opts: ResetOpts) {
    this.stopWatchdog();
    clearInterval(PromptHandler.runInterval);
    PromptHandler.runInterval = undefined;
    // PromptHandler spies on ui.setModeInternal; re-spying an existing spy would recurse forever.
    vi.restoreAllMocks();

    this.game = new GameManager(this.phaserGame);
    BattleScene.prototype.randBattleSeedInt = realRandBattleSeedInt;
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
      // Outside daily runs, movesets come from the starter-select screen; use its fresh-save default.
      // The test helper rolls these randomly before the run seed exists; pin them for reproducibility.
      for (const starter of starters) {
        Object.assign(starter, {
          ivs: new Array(6).fill(opts.ivs ?? 31),
          nature: opts.nature ?? Nature.HARDY,
          abilityIndex: 0,
          shiny: false,
          variant: 0,
          female: false,
        });
        if (!starter.moveset?.length) {
          starter.moveset = getStarterMoves(starter.speciesId, starter.formIndex).slice(0, 4) as StarterMoveset;
        }
      }
      // generateStarters() forces seed "test", so seed afterwards.
      this.scene.setSeed(opts.seed ?? randomString(24));
      this.scene.resetSeed();
      this.scene.phaseManager.pushNew("EncounterPhase", false);
      new SelectStarterPhase().initBattle(starters);
    });
    await this.game.phaseInterceptor.to("EncounterPhase");

    this.startWatchdog();
    return this.advance();
  }

  async step(action: number) {
    const phase = this.scene.phaseManager.getCurrentPhase() as CommandPhase;
    if (!phase?.is("CommandPhase")) {
      throw new Error(`step() called outside a CommandPhase (in ${phase?.phaseName})`);
    }
    const mask = this.legalActions(phase);
    let invalid = !mask[action];
    if (!invalid) {
      invalid = !this.issue(phase, action);
    }
    if (invalid) {
      // Fall back to the first legal action so the episode can continue.
      const fallback = mask.findIndex(Boolean);
      if (fallback < 0 || !this.issue(phase, fallback)) {
        throw new Error(`No legal action could be issued (requested ${action})`);
      }
    }
    const result = await this.advance();
    result.info.invalidAction = invalid;
    return result;
  }

  private issue(phase: CommandPhase, action: number): boolean {
    return action < 4
      ? phase.handleCommand(Command.FIGHT, action)
      : phase.handleCommand(Command.POKEMON, action - 4, false);
  }

  /** Run phases until the agent must act (CommandPhase awaiting input) or the run ends. */
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
        this.count(`skip:${name}`);
        pm.shiftPhase();
        continue;
      }

      await this.runPhase(phase);

      if (
        name === "CommandPhase"
        && this.interceptorState === "interrupted"
        && pm.getCurrentPhase() === phase
        && this.scene.ui.mode === UiMode.COMMAND
      ) {
        return this.result(false);
      }
    }
    throw new Error(`advance(): exceeded ${MAX_PHASES_PER_STEP} phases without a decision`);
  }

  /** Start a phase and wait until it ends or interrupts for UI input (no 50ms polling). */
  private async runPhase(phase: ReturnType<BattleScene["phaseManager"]["getCurrentPhase"]>) {
    const interceptor = this.game.phaseInterceptor;
    interceptor["target"] = phase.phaseName;
    interceptor["state"] = "running";
    phase.start();
    await this.waitUntil(() => this.interceptorState !== "running", `running ${phase.phaseName}`);
  }

  private async waitUntil(cond: () => boolean, what: string) {
    const start = Date.now();
    while (!cond()) {
      if (Date.now() - start > STALL_TIMEOUT_MS) {
        const phase = this.scene.phaseManager.getCurrentPhase()?.phaseName;
        throw new Error(`Stalled ${STALL_TIMEOUT_MS}ms while ${what} (phase=${phase}, ui=${UiMode[this.scene.ui.mode]})`);
      }
      await new Promise(resolve => setImmediate(resolve));
    }
  }

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
    if (!handler?.active || this.interceptorState === "idling") {
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

    switch (ui.mode) {
      case UiMode.COMMAND:
      case UiMode.FIGHT:
        // The agent's decision; never auto-answer.
        return;
      case UiMode.PARTY: {
        if (phaseName === "SwitchPhase") {
          const slot = this.firstBenchSlot();
          if (slot >= 0) {
            handler.setCursor(slot);
            handler.processInput(Button.ACTION); // select slot
            handler.processInput(Button.ACTION); // "Send out"
            this.count("faint_switch");
            return;
          }
        }
        handler.processInput(Button.CANCEL);
        this.count(`party_cancel:${phaseName}`);
        return;
      }
      case UiMode.MODIFIER_SELECT:
        if (handler.awaitingActionInput) {
          handler.processInput(Button.CANCEL); // skip rewards (confirmed below)
          this.count("skip_rewards");
        }
        return;
      case UiMode.CONFIRM:
        handler.processInput(phaseName === "SelectModifierPhase" ? Button.ACTION : Button.CANCEL);
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

  private firstBenchSlot(): number {
    const party = this.scene.getPlayerParty();
    const battlers = this.scene.currentBattle.getBattlerCount();
    return party.findIndex((p, i) => i >= battlers && p.isAllowedInBattle() && !p.isOnField());
  }

  private count(key: string) {
    this.autoInputs[key] = (this.autoInputs[key] ?? 0) + 1;
  }

  // #endregion

  // #region Observation

  private legalActions(phase: CommandPhase): boolean[] {
    const mask = new Array<boolean>(NUM_ACTIONS).fill(false);
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
    return mask;
  }

  private encodePokemon(p: Pokemon) {
    return {
      species: p.species.speciesId,
      level: p.level,
      hp: p.hp,
      maxHp: p.getMaxHp(),
      status: p.status?.effect ?? 0,
      types: p.getTypes(),
      fainted: p.isFainted(),
      onField: p.isOnField(),
      boss: p.isBoss(),
      moves: p.getMoveset().map(m => ({ id: m.moveId, pp: m.getMovePp() - m.ppUsed, maxPp: m.getMovePp() })),
    };
  }

  private result(done: boolean, reason?: string) {
    const battle = this.scene.currentBattle;
    const phase = this.scene.phaseManager.getCurrentPhase();
    const actor = !done && phase?.is("CommandPhase") ? (phase as CommandPhase).getFieldIndex() : -1;
    return {
      obs: {
        wave: battle?.waveIndex ?? 0,
        turn: battle?.turn ?? 0,
        double: battle?.double ?? false,
        actor,
        party: this.scene.getPlayerParty().map(p => this.encodePokemon(p)),
        enemies: this.scene.getEnemyField().map(p => this.encodePokemon(p)),
        legal: actor >= 0 ? this.legalActions(phase as CommandPhase) : new Array(NUM_ACTIONS).fill(false),
      },
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
