/**
 * Noesis — Dedicated error logger
 *
 * Writes structured error logs to ~/.openclaw/noesis/error.log
 * Independent of OpenClaw's api.logger — errors are never lost
 * even if OpenClaw's log stream is broken.
 */

import fs from "fs";
import path from "path";
import os from "os";

const DEFAULT_ERROR_LOG_PATH = "~/.openclaw/noesis/error.log";

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

let errorLogPath: string = resolvePath(DEFAULT_ERROR_LOG_PATH);
let rotationWarningLogged = false;

/**
 * Configure the error log path (call before any logging).
 */
export function configureErrorLog(logPath: string): void {
  errorLogPath = resolvePath(logPath);
  ensureLogDir();
}

/**
 * Write an error to the dedicated error log file.
 * Appends a single-line JSON entry per error.
 * Silently no-ops on write failures (don't crash over logging).
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
    ensureLogDir();

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

    checkRotation();
  } catch {
    // Logging should never crash the plugin
  }
}

/**
 * Serialize any rejection reason into a readable string.
 * Handles Error instances, strings, numbers, booleans, null, undefined,
 * and arbitrary objects (via JSON stringify fallback).
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
 * Same as logError but level=fatal for filtering.
 */
export function logFatal(
  message: string,
  error?: unknown,
  context?: Record<string, unknown>
): void {
  try {
    ensureLogDir();

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
      // Non-Error value: use serialized reason as message
      entry.message = serializeReason(error ?? message);
    }

    const line = JSON.stringify(entry) + "\n";
    fs.appendFileSync(errorLogPath, line, { encoding: "utf8" });

    checkRotation();
  } catch {
    // Logging should never crash the plugin
  }
}

/**
 * Install global unhandled rejection and uncaught exception handlers
 * that write to the noesis error log before propagating.
 * Call once during plugin init.
 */
export function installGlobalErrorHandlers(): void {
  process.on("unhandledRejection", (reason: unknown, promise: Promise<unknown>) => {
    const serialized = serializeReason(reason);
    logFatal("Unhandled Promise Rejection", reason, { context: "unhandledRejection", serialized });
  });

  process.on("uncaughtException", (error: Error) => {
    logFatal("Uncaught Exception", error, { context: "uncaughtException" });
    // Don't exit — let OpenClaw manage shutdown
  });
}

/**
 * Wrap an async function so any thrown error is logged to error.log
 * before propagating. Use for tool execute() callbacks.
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

function resolvePath(p: string): string {
  return p.replace(/^~/, os.homedir());
}

function ensureLogDir(): void {
  const dir = path.dirname(errorLogPath);
  if (!fs.existsSync(dir)) {
    fs.mkdirSync(dir, { recursive: true });
  }
}

/** Truncate error log if it grows beyond 5 MB. */
const MAX_LOG_BYTES = 5 * 1024 * 1024;

function checkRotation(): void {
  if (rotationWarningLogged) return;
  try {
    const stat = fs.statSync(errorLogPath);
    if (stat.size > MAX_LOG_BYTES) {
      // Truncate to last 1000 lines
      const content = fs.readFileSync(errorLogPath, "utf8");
      const lines = content.split("\n").filter(Boolean);
      const trimmed = lines.slice(-1000).join("\n") + "\n";
      fs.writeFileSync(errorLogPath, trimmed, { encoding: "utf8" });
      rotationWarningLogged = true;
    }
  } catch {
    // Ignore rotation errors
  }
}
