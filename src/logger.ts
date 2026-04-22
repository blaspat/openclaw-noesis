/**
 * Noesis — Dedicated file logger
 *
 * Writes all plugin logs to ~/.openclaw/noesis/noesis.log
 * and errors to ~/.openclaw/noesis/error.log.
 * Independent of OpenClaw's api.logger — no plugin output reaches the gateway log.
 */

import fs from "fs";
import path from "path";
import os from "os";

const DEFAULT_NOESIS_LOG_PATH = "~/.openclaw/noesis/noesis.log";
const DEFAULT_ERROR_LOG_PATH = "~/.openclaw/noesis/error.log";
const MAX_LOG_BYTES = 5 * 1024 * 1024;

let noesisLogPath: string = resolvePath(DEFAULT_NOESIS_LOG_PATH);
let errorLogPath: string = resolvePath(DEFAULT_ERROR_LOG_PATH);
let noesisLogRotationLogged = false;
let errorLogRotationLogged = false;

/**
 * Configure both log paths (call before any logging).
 */
export function configureLogs(noesisPath: string, errorPath: string): void {
  noesisLogPath = resolvePath(noesisPath);
  errorLogPath = resolvePath(errorPath);
  ensureLogDir(noesisLogPath);
  ensureLogDir(errorLogPath);
}

/**
 * Write an info/debug line to the noesis log file.
 * Silently no-ops on write failures.
 */
export function logInfo(message: string): void {
  try {
    ensureLogDir(noesisLogPath);
    const line = `${timestamp()} [INFO] ${message}\n`;
    fs.appendFileSync(noesisLogPath, line, { encoding: "utf8" });
    checkRotation(noesisLogPath, () => { noesisLogRotationLogged = true; });
  } catch {
    // Logging should never crash the plugin
  }
}

/**
 * Write an error to the dedicated error log file.
 * Appends a single-line JSON entry per error.
 * Silently no-ops on write failures.
 */
export function logError(
  message: string,
  context?: {
    error?: unknown;
    tool?: string;
    agentId?: string;
    sessionId?: string;
    extra?: Record<string, unknown>;
  }
): void {
  try {
    ensureLogDir(errorLogPath);

    const entry: ErrorLogEntry = {
      timestamp: new Date().toISOString(),
      level: "error",
      message,
      tool: context?.tool,
      agentId: context?.agentId,
      sessionId: context?.sessionId,
    };

    if (context?.error instanceof Error) {
      entry.stack = context.error.stack ?? context.error.message;
      entry.message = `${message}: ${context.error.message}`;
    } else if (context?.error !== undefined) {
      entry.message = `${message}: ${String(context.error)}`;
    }

    if (context?.extra) {
      entry.context = context.extra;
    }

    const line = JSON.stringify(entry) + "\n";
    fs.appendFileSync(errorLogPath, line, { encoding: "utf8" });
    checkRotation(errorLogPath, () => { errorLogRotationLogged = true; });
  } catch {
    // Logging should never crash the plugin
  }
}

/**
 * Serialize any rejection reason into a readable string.
 */
function serializeReason(reason: unknown): string {
  if (reason instanceof Error) return reason.message;
  if (typeof reason === "string") return reason;
  if (typeof reason === "number" || typeof reason === "boolean") return String(reason);
  if (reason === null) return "null";
  if (reason === undefined) return "undefined";
  try {
    return JSON.stringify(reason);
  } catch {
    return Object.prototype.toString.call(reason);
  }
}

/**
 * Write a fatal error (unhandled rejection, uncaught exception).
 */
export function logFatal(
  message: string,
  error?: unknown,
  context?: Record<string, unknown>
): void {
  try {
    ensureLogDir(errorLogPath);

    const entry: ErrorLogEntry = {
      timestamp: new Date().toISOString(),
      level: "fatal",
      message,
      tool: context?.tool as string | undefined,
      agentId: context?.agentId as string | undefined,
      sessionId: context?.sessionId as string | undefined,
    };

    if (error instanceof Error) {
      entry.stack = error.stack ?? error.message;
    } else {
      entry.message = serializeReason(error ?? message);
    }

    const line = JSON.stringify(entry) + "\n";
    fs.appendFileSync(errorLogPath, line, { encoding: "utf8" });
    checkRotation(errorLogPath, () => { errorLogRotationLogged = true; });
  } catch {
    // Logging should never crash the plugin
  }
}

/**
 * Install global unhandled rejection and uncaught exception handlers.
 */
export function installGlobalErrorHandlers(): void {
  process.on("unhandledRejection", (reason: unknown) => {
    logFatal("Unhandled Promise Rejection", reason, { context: "unhandledRejection" });
  });

  process.on("uncaughtException", (error: Error) => {
    logFatal("Uncaught Exception", error, { context: "uncaughtException" });
  });
}

/**
 * Wrap an async function so any thrown error is logged before propagating.
 */
export function withErrorLog<T extends (...args: any[]) => Promise<any>>(
  fn: T,
  context?: { tool?: string; agentId?: string; sessionId?: string }
): T {
  return ((...args: Parameters<T>) => fn(...args).catch((err) => {
    logError(`Unhandled error in ${context?.tool ?? "unknown"}`, {
      error: err,
      tool: context?.tool,
      agentId: context?.agentId,
      sessionId: context?.sessionId,
    });
    throw err;
  })) as T;
}

// ─── internals ────────────────────────────────────────────────────────────────

function timestamp(): string {
  return new Date().toISOString();
}

function resolvePath(p: string): string {
  return p.replace(/^~/, os.homedir());
}

function ensureLogDir(logPath: string): void {
  const dir = path.dirname(logPath);
  if (!fs.existsSync(dir)) {
    fs.mkdirSync(dir, { recursive: true });
  }
}

/**
 * Truncate a log file to the last 1000 lines if it exceeds MAX_LOG_BYTES.
 * Calls onRotation only once per process lifetime (prevents spam on large logs).
 */
function checkRotation(logPath: string, onRotated: () => void): void {
  try {
    const stat = fs.statSync(logPath);
    if (stat.size > MAX_LOG_BYTES) {
      const content = fs.readFileSync(logPath, "utf8");
      const lines = content.split("\n").filter(Boolean);
      const trimmed = lines.slice(-1000).join("\n") + "\n";
      fs.writeFileSync(logPath, trimmed, { encoding: "utf8" });
      onRotated();
    }
  } catch {
    // Ignore rotation errors
  }
}

// ─── types ────────────────────────────────────────────────────────────────────

export interface ErrorLogEntry {
  timestamp: string;
  level: "error" | "warn" | "fatal";
  message: string;
  stack?: string;
  context?: Record<string, unknown>;
  tool?: string;
  agentId?: string;
  sessionId?: string;
}
