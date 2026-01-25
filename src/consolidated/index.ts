import process from "process";
import { McpServer, ResourceTemplate } from "@modelcontextprotocol/sdk/server/mcp.js";
import { StdioServerTransport } from "@modelcontextprotocol/sdk/server/stdio.js";
import { SSEServerTransport } from "@modelcontextprotocol/sdk/server/sse.js";
import express from "express";
import cors from "cors";
import {
  RootsListChangedNotificationSchema,
  type Root,
  type CallToolResult,
  type Resource,
  type ResourceLink,
  ElicitResultSchema,
  type CreateMessageRequest,
  CreateMessageResultSchema,
  type GetTaskResult,
  type Task,
  type ElicitResult,
} from "@modelcontextprotocol/sdk/types.js";
import {
  InMemoryTaskStore,
  InMemoryTaskMessageQueue,
  type CreateTaskResult,
} from "@modelcontextprotocol/sdk/experimental/tasks/index.js";
import { completable } from "@modelcontextprotocol/sdk/server/completable.js";
import { z } from "zod";
import fs from "fs/promises";
import { createReadStream, Dirent, readFileSync, readdirSync, statSync } from "fs";
import path from "path";
import os from 'os';
import { fileURLToPath } from 'url';
import chalk from 'chalk';
import { randomBytes } from 'crypto';
import { createTwoFilesPatch } from 'diff';
import { minimatch } from 'minimatch';
import { exec } from 'child_process';
import { promisify } from 'util';
import { gzipSync } from "node:zlib";

const execAsync = promisify(exec);

// --- Path Utilities ---
function convertToWindowsPath(p: string): string {
  if (p.startsWith('/mnt/')) return p;
  if (p.match(/^\/[a-zA-Z]\//) && process.platform === 'win32') {
    const driveLetter = p.charAt(1).toUpperCase();
    const pathPart = p.slice(2).replace(/\//g, '\\');
    return `${driveLetter}:${pathPart}`;
  }
  if (p.match(/^[a-zA-Z]:/)) return p.replace(/\//g, '\\');
  return p;
}

function normalizePath(p: string): string {
  p = p.trim().replace(/^["']|["']$/g, '');
  const isUnixPath = p.startsWith('/') && (
    p.match(/^\/mnt\/[a-z]\//i) ||
    (process.platform !== 'win32') ||
    (process.platform === 'win32' && !p.match(/^\/[a-zA-Z]\//))
  );
  if (isUnixPath) return p.replace(/\/+/g, '/').replace(/(?<!^)\/$/, '');
  p = convertToWindowsPath(p);
  if (p.startsWith('\\\\')) {
    let uncPath = p.replace(/^\\{2,}/, '\\\\');
    const restOfPath = uncPath.substring(2).replace(/\\\\/g, '\\');
    p = '\\\\' + restOfPath;
  } else {
    p = p.replace(/\\\\/g, '\\');
  }
  let normalized = path.normalize(p);
  if (p.startsWith('\\\\') && !normalized.startsWith('\\\\')) normalized = '\\' + normalized;
  if (normalized.match(/^[a-zA-Z]:/)) {
    let result = normalized.replace(/\//g, '\\');
    if (/^[a-z]:/.test(result)) result = result.charAt(0).toUpperCase() + result.slice(1);
    return result;
  }
  if (process.platform === 'win32') return normalized.replace(/\//g, '\\');
  return normalized;
}

function expandHome(filepath: string): string {
  if (filepath.startsWith('~/') || filepath === '~') return path.join(os.homedir(), filepath.slice(1));
  return filepath;
}

function isPathWithinAllowedDirectories(absolutePath: string, allowedDirectories: string[]): boolean {
  if (typeof absolutePath !== 'string' || !Array.isArray(allowedDirectories)) return false;
  if (!absolutePath || allowedDirectories.length === 0) return false;
  if (absolutePath.includes('\x00')) return false;
  let normalizedPath: string;
  try {
    normalizedPath = path.resolve(path.normalize(absolutePath));
  } catch {
    return false;
  }
  if (!path.isAbsolute(normalizedPath)) throw new Error('Path must be absolute after normalization');
  return allowedDirectories.some(dir => {
    if (typeof dir !== 'string' || !dir) return false;
    if (dir.includes('\x00')) return false;
    let normalizedDir: string;
    try {
      normalizedDir = path.resolve(path.normalize(dir));
    } catch {
      return false;
    }
    if (!path.isAbsolute(normalizedDir)) throw new Error('Allowed directories must be absolute paths after normalization');
    if (normalizedPath === normalizedDir) return true;
    if (normalizedDir === path.sep) return normalizedPath.startsWith(path.sep);
    if (path.sep === '\\' && normalizedDir.match(/^[A-Za-z]:\\?$/)) {
      const dirDrive = normalizedDir.charAt(0).toLowerCase();
      const pathDrive = normalizedPath.charAt(0).toLowerCase();
      return pathDrive === dirDrive && normalizedPath.startsWith(normalizedDir.replace(/\\?$/, '\\'));
    }
    return normalizedPath.startsWith(normalizedDir + path.sep);
  });
}

// --- Memory Server Logic ---
export interface Entity {
  name: string;
  entityType: string;
  observations: string[];
}

export interface Relation {
  from: string;
  to: string;
  relationType: string;
}

export interface KnowledgeGraph {
  entities: Entity[];
  relations: Relation[];
}

class KnowledgeGraphManager {
  constructor(private memoryFilePath: string) {}

  private async loadGraph(): Promise<KnowledgeGraph> {
    try {
      const data = await fs.readFile(this.memoryFilePath, "utf-8");
      const lines = data.split("\n").filter(line => line.trim() !== "");
      return lines.reduce((graph: KnowledgeGraph, line) => {
        const item = JSON.parse(line);
        if (item.type === "entity") {
          graph.entities.push({
            name: item.name,
            entityType: item.entityType,
            observations: item.observations
          });
        }
        if (item.type === "relation") {
          graph.relations.push({
            from: item.from,
            to: item.to,
            relationType: item.relationType
          });
        }
        return graph;
      }, { entities: [], relations: [] });
    } catch (error) {
      if (error instanceof Error && (error as any).code === "ENOENT") {
        return { entities: [], relations: [] };
      }
      throw error;
    }
  }

  private async saveGraph(graph: KnowledgeGraph): Promise<void> {
    const lines = [
      ...graph.entities.map(e => JSON.stringify({
        type: "entity",
        name: e.name,
        entityType: e.entityType,
        observations: e.observations
      })),
      ...graph.relations.map(r => JSON.stringify({
        type: "relation",
        from: r.from,
        to: r.to,
        relationType: r.relationType
      })),
    ];
    await fs.writeFile(this.memoryFilePath, lines.join("\n"));
  }

  async createEntities(entities: Entity[]): Promise<Entity[]> {
    const graph = await this.loadGraph();
    const newEntities = entities.filter(e => !graph.entities.some(existingEntity => existingEntity.name === e.name));
    graph.entities.push(...newEntities);
    await this.saveGraph(graph);
    return newEntities;
  }

  async createRelations(relations: Relation[]): Promise<Relation[]> {
    const graph = await this.loadGraph();
    const newRelations = relations.filter(r => !graph.relations.some(existingRelation => 
      existingRelation.from === r.from && 
      existingRelation.to === r.to && 
      existingRelation.relationType === r.relationType
    ));
    graph.relations.push(...newRelations);
    await this.saveGraph(graph);
    return newRelations;
  }

  async addObservations(observations: { entityName: string; contents: string[] }[]): Promise<{ entityName: string; addedObservations: string[] }[]> {
    const graph = await this.loadGraph();
    const results = observations.map(o => {
      const entity = graph.entities.find(e => e.name === o.entityName);
      if (!entity) {
        throw new Error(`Entity with name ${o.entityName} not found`);
      }
      const newObservations = o.contents.filter(content => !entity.observations.includes(content));
      entity.observations.push(...newObservations);
      return { entityName: o.entityName, addedObservations: newObservations };
    });
    await this.saveGraph(graph);
    return results;
  }

  async deleteEntities(entityNames: string[]): Promise<void> {
    const graph = await this.loadGraph();
    graph.entities = graph.entities.filter(e => !entityNames.includes(e.name));
    graph.relations = graph.relations.filter(r => !entityNames.includes(r.from) && !entityNames.includes(r.to));
    await this.saveGraph(graph);
  }

  async deleteObservations(deletions: { entityName: string; observations: string[] }[]): Promise<void> {
    const graph = await this.loadGraph();
    deletions.forEach(d => {
      const entity = graph.entities.find(e => e.name === d.entityName);
      if (entity) {
        entity.observations = entity.observations.filter(o => !d.observations.includes(o));
      }
    });
    await this.saveGraph(graph);
  }

  async deleteRelations(relations: Relation[]): Promise<void> {
    const graph = await this.loadGraph();
    graph.relations = graph.relations.filter(r => !relations.some(delRelation => 
      r.from === delRelation.from && 
      r.to === delRelation.to && 
      r.relationType === delRelation.relationType
    ));
    await this.saveGraph(graph);
  }

  async readGraph(): Promise<KnowledgeGraph> {
    return this.loadGraph();
  }

  async searchNodes(query: string): Promise<KnowledgeGraph> {
    const graph = await this.loadGraph();
    const filteredEntities = graph.entities.filter(e => 
      e.name.toLowerCase().includes(query.toLowerCase()) ||
      e.entityType.toLowerCase().includes(query.toLowerCase()) ||
      e.observations.some(o => o.toLowerCase().includes(query.toLowerCase()))
    );
    const filteredEntityNames = new Set(filteredEntities.map(e => e.name));
    const filteredRelations = graph.relations.filter(r => 
      filteredEntityNames.has(r.from) && filteredEntityNames.has(r.to)
    );
    return { entities: filteredEntities, relations: filteredRelations };
  }

  async openNodes(names: string[]): Promise<KnowledgeGraph> {
    const graph = await this.loadGraph();
    const filteredEntities = graph.entities.filter(e => names.includes(e.name));
    const filteredEntityNames = new Set(filteredEntities.map(e => e.name));
    const filteredRelations = graph.relations.filter(r => 
      filteredEntityNames.has(r.from) && filteredEntityNames.has(r.to)
    );
    return { entities: filteredEntities, relations: filteredRelations };
  }
}

// --- Sequential Thinking Logic ---
export interface ThoughtData {
  thought: string;
  thoughtNumber: number;
  totalThoughts: number;
  isRevision?: boolean;
  revisesThought?: number;
  branchFromThought?: number;
  branchId?: string;
  needsMoreThoughts?: boolean;
  nextThoughtNeeded: boolean;
}

class SequentialThinkingServer {
  private thoughtHistory: ThoughtData[] = [];
  private branches: Record<string, ThoughtData[]> = {};
  private disableThoughtLogging: boolean;

  constructor() {
    this.disableThoughtLogging = (process.env.DISABLE_THOUGHT_LOGGING || "").toLowerCase() === "true";
  }

  private formatThought(thoughtData: ThoughtData): string {
    const { thoughtNumber, totalThoughts, thought, isRevision, revisesThought, branchFromThought, branchId } = thoughtData;
    let prefix = '';
    let context = '';
    if (isRevision) {
      prefix = chalk.yellow('🔄 Revision');
      context = ` (revising thought ${revisesThought})`;
    } else if (branchFromThought) {
      prefix = chalk.green('🌿 Branch');
      context = ` (from thought ${branchFromThought}, ID: ${branchId})`;
    } else {
      prefix = chalk.blue('💭 Thought');
      context = '';
    }
    const header = `${prefix} ${thoughtNumber}/${totalThoughts}${context}`;
    const border = '─'.repeat(Math.max(header.length, thought.length) + 4);
    return `\n┌${border}┐\n│ ${header} │\n├${border}┤\n│ ${thought.padEnd(border.length - 2)} │\n└${border}┘`;
  }

  public processThought(input: ThoughtData) {
    if (input.thoughtNumber > input.totalThoughts) {
      input.totalThoughts = input.thoughtNumber;
    }
    this.thoughtHistory.push(input);
    if (input.branchFromThought && input.branchId) {
      if (!this.branches[input.branchId]) {
        this.branches[input.branchId] = [];
      }
      this.branches[input.branchId].push(input);
    }
    if (!this.disableThoughtLogging) {
      console.error(this.formatThought(input));
    }
    return {
      thoughtNumber: input.thoughtNumber,
      totalThoughts: input.totalThoughts,
      nextThoughtNeeded: input.nextThoughtNeeded,
      branches: Object.keys(this.branches),
      thoughtHistoryLength: this.thoughtHistory.length
    };
  }
}

// --- Filesystem Logic ---
let allowedDirectories: string[] = [];

function setAllowedDirectories(directories: string[]): void {
  allowedDirectories = [...directories];
}

async function validatePath(requestedPath: string): Promise<string> {
  const expandedPath = expandHome(requestedPath);
  const absolute = path.isAbsolute(expandedPath)
    ? path.resolve(expandedPath)
    : path.resolve(process.cwd(), expandedPath);
  const normalizedRequested = normalizePath(absolute);
  if (!isPathWithinAllowedDirectories(normalizedRequested, allowedDirectories)) {
    throw new Error(`Access denied - path outside allowed directories: ${absolute}`);
  }
  try {
    const realPath = await fs.realpath(absolute);
    const normalizedReal = normalizePath(realPath);
    if (!isPathWithinAllowedDirectories(normalizedReal, allowedDirectories)) {
      throw new Error(`Access denied - symlink target outside allowed directories: ${realPath}`);
    }
    return realPath;
  } catch (error) {
    if ((error as any).code === 'ENOENT') {
      const parentDir = path.dirname(absolute);
      try {
        const realParentPath = await fs.realpath(parentDir);
        const normalizedParent = normalizePath(realParentPath);
        if (!isPathWithinAllowedDirectories(normalizedParent, allowedDirectories)) {
          throw new Error(`Access denied - parent directory outside allowed directories: ${realParentPath}`);
        }
        return absolute;
      } catch {
        throw new Error(`Parent directory does not exist: ${parentDir}`);
      }
    }
    throw error;
  }
}

async function readFileContent(filePath: string): Promise<string> {
  return await fs.readFile(filePath, 'utf-8');
}

async function writeFileContent(filePath: string, content: string): Promise<void> {
  try {
    await fs.writeFile(filePath, content, { encoding: "utf-8", flag: 'wx' });
  } catch (error) {
    if ((error as any).code === 'EEXIST') {
      const tempPath = `${filePath}.${randomBytes(16).toString('hex')}.tmp`;
      try {
        await fs.writeFile(tempPath, content, 'utf-8');
        await fs.rename(tempPath, filePath);
      } catch (renameError) {
        try { await fs.unlink(tempPath); } catch {}
        throw renameError;
      }
    } else {
      throw error;
    }
  }
}

function normalizeLineEndings(text: string): string {
  return text.replace(/\r\n/g, '\n');
}

async function tailFile(filePath: string, numLines: number): Promise<string> {
  const CHUNK_SIZE = 1024;
  const stats = await fs.stat(filePath);
  const fileSize = stats.size;
  if (fileSize === 0) return '';
  const fileHandle = await fs.open(filePath, 'r');
  try {
    const lines: string[] = [];
    let position = fileSize;
    let chunk = Buffer.alloc(CHUNK_SIZE);
    let linesFound = 0;
    let remainingText = '';
    while (position > 0 && linesFound < numLines) {
      const size = Math.min(CHUNK_SIZE, position);
      position -= size;
      const { bytesRead } = await fileHandle.read(chunk, 0, size, position);
      if (!bytesRead) break;
      const readData = chunk.slice(0, bytesRead).toString('utf-8');
      const chunkText = readData + remainingText;
      const chunkLines = normalizeLineEndings(chunkText).split('\n');
      if (position > 0) {
        remainingText = chunkLines[0];
        chunkLines.shift();
      }
      for (let i = chunkLines.length - 1; i >= 0 && linesFound < numLines; i--) {
        lines.unshift(chunkLines[i]);
        linesFound++;
      }
    }
    return lines.join('\n');
  } finally {
    await fileHandle.close();
  }
}

async function headFile(filePath: string, numLines: number): Promise<string> {
  const fileHandle = await fs.open(filePath, 'r');
  try {
    const lines: string[] = [];
    let buffer = '';
    let bytesRead = 0;
    const chunk = Buffer.alloc(1024);
    while (lines.length < numLines) {
      const result = await fileHandle.read(chunk, 0, chunk.length, bytesRead);
      if (result.bytesRead === 0) break;
      bytesRead += result.bytesRead;
      buffer += chunk.slice(0, result.bytesRead).toString('utf-8');
      const newLineIndex = buffer.lastIndexOf('\n');
      if (newLineIndex !== -1) {
        const completeLines = buffer.slice(0, newLineIndex).split('\n');
        buffer = buffer.slice(newLineIndex + 1);
        for (const line of completeLines) {
          lines.push(line);
          if (lines.length >= numLines) break;
        }
      }
    }
    if (buffer.length > 0 && lines.length < numLines) lines.push(buffer);
    return lines.join('\n');
  } finally {
    await fileHandle.close();
  }
}

async function applyFileEdits(filePath: string, edits: { oldText: string; newText: string }[], dryRun: boolean): Promise<string> {
  const content = normalizeLineEndings(await fs.readFile(filePath, 'utf-8'));
  let modifiedContent = content;
  for (const edit of edits) {
    const normalizedOld = normalizeLineEndings(edit.oldText);
    const normalizedNew = normalizeLineEndings(edit.newText);
    if (modifiedContent.includes(normalizedOld)) {
      modifiedContent = modifiedContent.replace(normalizedOld, normalizedNew);
      continue;
    }
    const oldLines = normalizedOld.split('\n');
    const contentLines = modifiedContent.split('\n');
    let matchFound = false;
    for (let i = 0; i <= contentLines.length - oldLines.length; i++) {
      const potentialMatch = contentLines.slice(i, i + oldLines.length);
      const isMatch = oldLines.every((oldLine, j) => oldLine.trim() === potentialMatch[j].trim());
      if (isMatch) {
        const originalIndent = contentLines[i].match(/^\s*/)?.[0] || '';
        const newLines = normalizedNew.split('\n').map((line, j) => {
          if (j === 0) return originalIndent + line.trimStart();
          const oldIndent = oldLines[j]?.match(/^\s*/)?.[0] || '';
          const newIndent = line.match(/^\s*/)?.[0] || '';
          if (oldIndent && newIndent) {
            const relativeIndent = newIndent.length - oldIndent.length;
            return originalIndent + ' '.repeat(Math.max(0, relativeIndent)) + line.trimStart();
          }
          return line;
        });
        contentLines.splice(i, oldLines.length, ...newLines);
        modifiedContent = contentLines.join('\n');
        matchFound = true;
        break;
      }
    }
    if (!matchFound) throw new Error(`Could not find exact match for edit:\n${edit.oldText}`);
  }
  const diff = createTwoFilesPatch(filePath, filePath, content, modifiedContent);
  if (!dryRun) {
    const tempPath = `${filePath}.${randomBytes(16).toString('hex')}.tmp`;
    await fs.writeFile(tempPath, modifiedContent, 'utf-8');
    await fs.rename(tempPath, filePath);
  }
  return diff;
}

async function readFileAsBase64Stream(filePath: string): Promise<string> {
  return new Promise((resolve, reject) => {
    const stream = createReadStream(filePath);
    const chunks: Buffer[] = [];
    stream.on('data', (chunk) => chunks.push(chunk as Buffer));
    stream.on('end', () => resolve(Buffer.concat(chunks).toString('base64')));
    stream.on('error', (err) => reject(err));
  });
}

function formatSize(bytes: number): string {
  const units = ['B', 'KB', 'MB', 'GB', 'TB'];
  let size = bytes;
  let unitIndex = 0;
  while (size >= 1024 && unitIndex < units.length - 1) {
    size /= 1024;
    unitIndex++;
  }
  return `${size.toFixed(1)} ${units[unitIndex]}`;
}

// --- Main Server Initialization ---
function createMcpServer() {
  const taskStore = new InMemoryTaskStore();
  const taskMessageQueue = new InMemoryTaskMessageQueue();

  const server = new McpServer(
    {
      name: "consolidated-mcp-server",
      version: "1.0.0",
    },
    {
      capabilities: {
        tools: { listChanged: true },
        prompts: { listChanged: true },
        resources: { subscribe: true, listChanged: true },
        logging: {},
        tasks: {
          list: {},
          cancel: {},
          requests: { tools: { call: {} } },
        },
      },
      taskStore,
      taskMessageQueue,
    }
  );

  // Initialize Managers
  const memoryPath = path.join(path.dirname(fileURLToPath(import.meta.url)), 'memory.jsonl');
  const knowledgeGraphManager = new KnowledgeGraphManager(memoryPath);
  const thinkingServer = new SequentialThinkingServer();

  return { server, knowledgeGraphManager, thinkingServer, taskStore };
}

const { server, knowledgeGraphManager, thinkingServer, taskStore } = createMcpServer();

// Register Memory Tools
const EntitySchema = z.object({
  name: z.string().describe("The name of the entity"),
  entityType: z.string().describe("The type of the entity"),
  observations: z.array(z.string()).describe("An array of observation contents associated with the entity")
});

const RelationSchema = z.object({
  from: z.string().describe("The name of the entity where the relation starts"),
  to: z.string().describe("The name of the entity where the relation ends"),
  relationType: z.string().describe("The type of the relation")
});

server.registerTool("create_entities", { inputSchema: z.object({ entities: z.array(EntitySchema) }) }, async (args: any) => {
  const result = await knowledgeGraphManager.createEntities(args.entities);
  return { content: [{ type: "text", text: JSON.stringify(result, null, 2) }] };
});

server.registerTool("create_relations", { inputSchema: z.object({ relations: z.array(RelationSchema) }) }, async (args: any) => {
  const result = await knowledgeGraphManager.createRelations(args.relations);
  return { content: [{ type: "text", text: JSON.stringify(result, null, 2) }] };
});

server.registerTool("add_observations", {
  inputSchema: z.object({
    observations: z.array(z.object({
      entityName: z.string(),
      contents: z.array(z.string())
    }))
  })
}, async (args: any) => {
  const result = await knowledgeGraphManager.addObservations(args.observations);
  return { content: [{ type: "text", text: JSON.stringify(result, null, 2) }] };
});

server.registerTool("delete_entities", { inputSchema: z.object({ entityNames: z.array(z.string()) }) }, async (args: any) => {
  await knowledgeGraphManager.deleteEntities(args.entityNames);
  return { content: [{ type: "text", text: "Entities deleted successfully" }] };
});

server.registerTool("delete_observations", {
  inputSchema: z.object({
    deletions: z.array(z.object({
      entityName: z.string(),
      observations: z.array(z.string())
    }))
  })
}, async (args: any) => {
  await knowledgeGraphManager.deleteObservations(args.deletions);
  return { content: [{ type: "text", text: "Observations deleted successfully" }] };
});

server.registerTool("delete_relations", { inputSchema: z.object({ relations: z.array(RelationSchema) }) }, async (args: any) => {
  await knowledgeGraphManager.deleteRelations(args.relations);
  return { content: [{ type: "text", text: "Relations deleted successfully" }] };
});

server.registerTool("read_graph", { inputSchema: z.object({}) }, async () => {
  const graph = await knowledgeGraphManager.readGraph();
  return { content: [{ type: "text", text: JSON.stringify(graph, null, 2) }] };
});

server.registerTool("search_nodes", { inputSchema: z.object({ query: z.string() }) }, async (args: any) => {
  const graph = await knowledgeGraphManager.searchNodes(args.query);
  return { content: [{ type: "text", text: JSON.stringify(graph, null, 2) }] };
});

server.registerTool("open_nodes", { inputSchema: z.object({ names: z.array(z.string()) }) }, async (args: any) => {
  const graph = await knowledgeGraphManager.openNodes(args.names);
  return { content: [{ type: "text", text: JSON.stringify(graph, null, 2) }] };
});

// Register Sequential Thinking Tool
server.registerTool("sequentialthinking", {
  inputSchema: z.object({
    thought: z.string(),
    nextThoughtNeeded: z.boolean(),
    thoughtNumber: z.number().int().min(1),
    totalThoughts: z.number().int().min(1),
    isRevision: z.boolean().optional(),
    revisesThought: z.number().int().min(1).optional(),
    branchFromThought: z.number().int().min(1).optional(),
    branchId: z.string().optional(),
    needsMoreThoughts: z.boolean().optional()
  })
}, async (args: any) => {
  const result = thinkingServer.processThought(args);
  return { content: [{ type: "text", text: JSON.stringify(result, null, 2) }] };
});

// Register Filesystem Tools
server.registerTool("read_text_file", {
  inputSchema: z.object({
    path: z.string(),
    tail: z.number().optional(),
    head: z.number().optional()
  })
}, async (args: any) => {
  const validPath = await validatePath(args.path);
  let content: string;
  if (args.tail) {
    content = await tailFile(validPath, args.tail);
  } else if (args.head) {
    content = await headFile(validPath, args.head);
  } else {
    content = await readFileContent(validPath);
  }
  return { content: [{ type: "text", text: content }] };
});

server.registerTool("read_media_file", { inputSchema: z.object({ path: z.string() }) }, async (args: any) => {
  const validPath = await validatePath(args.path);
  const extension = path.extname(validPath).toLowerCase();
  const mimeTypes: Record<string, string> = {
    ".png": "image/png", ".jpg": "image/jpeg", ".jpeg": "image/jpeg", ".gif": "image/gif",
    ".webp": "image/webp", ".bmp": "image/bmp", ".svg": "image/svg+xml", ".mp3": "audio/mpeg",
    ".wav": "audio/wav", ".ogg": "audio/ogg", ".flac": "audio/flac",
  };
  const mimeType = mimeTypes[extension] || "application/octet-stream";
  const data = await readFileAsBase64Stream(validPath);
  return {
    content: [{
      type: mimeType.startsWith("image/") ? "image" : mimeType.startsWith("audio/") ? "audio" : "blob",
      data,
      mimeType
    } as any]
  };
});

server.registerTool("read_multiple_files", { inputSchema: z.object({ paths: z.array(z.string()) }) }, async (args: any) => {
  const results = await Promise.all(args.paths.map(async (p: string) => {
    try {
      const validPath = await validatePath(p);
      const content = await readFileContent(validPath);
      return `${p}:\n${content}\n`;
    } catch (e) {
      return `${p}: Error - ${e}\n`;
    }
  }));
  return { content: [{ type: "text", text: results.join("\n---\n") }] };
});

server.registerTool("write_file", { inputSchema: z.object({ path: z.string(), content: z.string() }) }, async (args: any) => {
  const validPath = await validatePath(args.path);
  await writeFileContent(validPath, args.content);
  return { content: [{ type: "text", text: `Successfully wrote to ${args.path}` }] };
});

server.registerTool("edit_file", {
  inputSchema: z.object({
    path: z.string(),
    edits: z.array(z.object({ oldText: z.string(), newText: z.string() })),
    dryRun: z.boolean().default(false)
  })
}, async (args: any) => {
  const validPath = await validatePath(args.path);
  const diff = await applyFileEdits(validPath, args.edits, args.dryRun);
  return { content: [{ type: "text", text: diff }] };
});

server.registerTool("create_directory", { inputSchema: z.object({ path: z.string() }) }, async (args: any) => {
  const validPath = await validatePath(args.path);
  await fs.mkdir(validPath, { recursive: true });
  return { content: [{ type: "text", text: `Successfully created directory ${args.path}` }] };
});

server.registerTool("list_directory", { inputSchema: z.object({ path: z.string() }) }, async (args: any) => {
  const validPath = await validatePath(args.path);
  const entries = await fs.readdir(validPath, { withFileTypes: true });
  const formatted = entries
    .map((entry: Dirent) => `${entry.isDirectory() ? "[DIR]" : "[FILE]"} ${entry.name}`)
    .join("\n");
  return { content: [{ type: "text", text: formatted }] };
});

server.registerTool("list_directory_with_sizes", {
  inputSchema: z.object({
    path: z.string(),
    sortBy: z.enum(['name', 'size']).optional().default('name')
  })
}, async (args: any) => {
  const validPath = await validatePath(args.path);
  const entries = await fs.readdir(validPath, { withFileTypes: true });
  const detailed = await Promise.all(entries.map(async (e: Dirent) => {
    const stats = await fs.stat(path.join(validPath, e.name));
    return { name: e.name, isDir: e.isDirectory(), size: stats.size };
  }));
  if (args.sortBy === 'size') detailed.sort((a, b) => b.size - a.size);
  else detailed.sort((a, b) => a.name.localeCompare(b.name));
  const formatted = detailed.map(e => `${e.isDir ? "[DIR]" : "[FILE]"} ${e.name.padEnd(30)} ${e.isDir ? "" : formatSize(e.size).padStart(10)}`).join("\n");
  return { content: [{ type: "text", text: formatted }] };
});

server.registerTool("directory_tree", {
  inputSchema: z.object({
    path: z.string(),
    excludePatterns: z.array(z.string()).optional().default([])
  })
}, async (args: any) => {
  const rootPath = await validatePath(args.path);
  async function buildTree(currentPath: string): Promise<any> {
    const entries = await fs.readdir(currentPath, { withFileTypes: true });
    const result: any[] = [];
    for (const entry of entries as Dirent[]) {
      const relPath = path.relative(rootPath, path.join(currentPath, entry.name));
      if (args.excludePatterns.some((p: string) => minimatch(relPath, p, { dot: true }))) continue;
      const node: any = { name: entry.name, type: entry.isDirectory() ? 'directory' : 'file' };
      if (entry.isDirectory()) node.children = await buildTree(path.join(currentPath, entry.name));
      result.push(node);
    }
    return result;
  }
  const tree = await buildTree(rootPath);
  return { content: [{ type: "text", text: JSON.stringify(tree, null, 2) }] };
});

server.registerTool("move_file", { inputSchema: z.object({ source: z.string(), destination: z.string() }) }, async (args: any) => {
  const validSource = await validatePath(args.source);
  const validDest = await validatePath(args.destination);
  await fs.rename(validSource, validDest);
  return { content: [{ type: "text", text: `Successfully moved ${args.source} to ${args.destination}` }] };
});

server.registerTool("search_files", {
  inputSchema: z.object({
    path: z.string(),
    pattern: z.string(),
    excludePatterns: z.array(z.string()).optional().default([])
  })
}, async (args: any) => {
  const rootPath = await validatePath(args.path);
  const results: string[] = [];
  async function search(currentPath: string) {
    const entries = await fs.readdir(currentPath, { withFileTypes: true });
    for (const entry of entries as Dirent[]) {
      const fullPath = path.join(currentPath, entry.name);
      const relPath = path.relative(rootPath, fullPath);
      if (args.excludePatterns.some((p: string) => minimatch(relPath, p, { dot: true }))) continue;
      if (minimatch(relPath, args.pattern, { dot: true })) results.push(fullPath);
      if (entry.isDirectory()) await search(fullPath);
    }
  }
  await search(rootPath);
  return { content: [{ type: "text", text: results.join("\n") || "No matches found" }] };
});

server.registerTool("get_file_info", { inputSchema: z.object({ path: z.string() }) }, async (args: any) => {
  const validPath = await validatePath(args.path);
  const stats = await fs.stat(validPath);
  const info = {
    size: stats.size,
    created: stats.birthtime,
    modified: stats.mtime,
    isDirectory: stats.isDirectory(),
    isFile: stats.isFile(),
    permissions: stats.mode.toString(8).slice(-3)
  };
  return { content: [{ type: "text", text: JSON.stringify(info, null, 2) }] };
});

server.registerTool("list_allowed_directories", { inputSchema: z.object({}) }, async () => {
  return { content: [{ type: "text", text: `Allowed directories:\n${allowedDirectories.join('\n')}` }] };
});

// --- Time Tools ---
server.registerTool("get_current_time", { inputSchema: z.object({ timezone: z.string() }) }, async (args: any) => {
  const now = new Date();
  const timeStr = now.toLocaleString("en-US", { timeZone: args.timezone });
  return { content: [{ type: "text", text: `Current time in ${args.timezone}: ${timeStr}` }] };
});

// --- Fetch Tools ---
server.registerTool("fetch", { inputSchema: z.object({ url: z.string() }) }, async (args: any) => {
  try {
    const response = await fetch(args.url);
    const text = await response.text();
    return { content: [{ type: "text", text: text.slice(0, 5000) }] };
  } catch (error) {
    return { content: [{ type: "text", text: `Error fetching ${args.url}: ${error}` }] };
  }
});

// --- Git Tools ---
server.registerTool("git_status", { inputSchema: z.object({ repo_path: z.string() }) }, async (args: any) => {
  try {
    const { stdout } = await execAsync(`git -C "${args.repo_path}" status`);
    return { content: [{ type: "text", text: stdout }] };
  } catch (error) {
    return { content: [{ type: "text", text: `Error running git status: ${error}` }] };
  }
});

server.registerTool("git_log", { inputSchema: z.object({ repo_path: z.string(), max_count: z.number().optional().default(10) }) }, async (args: any) => {
  try {
    const { stdout } = await execAsync(`git -C "${args.repo_path}" log -n ${args.max_count}`);
    return { content: [{ type: "text", text: stdout }] };
  } catch (error) {
    return { content: [{ type: "text", text: `Error running git log: ${error}` }] };
  }
});

// --- Everything Logic ---
const logsUpdateIntervals = new Map<string | undefined, NodeJS.Timeout>();
const loggingClients = new Set<string | undefined>();
const subscriberClients = new Set<string | undefined>();
const subscriberIntervals = new Map<string | undefined, NodeJS.Timeout>();

const RESOURCE_TYPE_TEXT = "Text" as const;
const RESOURCE_TYPE_BLOB = "Blob" as const;
const RESOURCE_TYPES = [RESOURCE_TYPE_TEXT, RESOURCE_TYPE_BLOB];

const textUriBase = "demo://resource/dynamic/text";
const blobUriBase = "demo://resource/dynamic/blob";

const textResourceUri = (resourceId: number) => new URL(`${textUriBase}/${resourceId}`);
const blobResourceUri = (resourceId: number) => new URL(`${blobUriBase}/${resourceId}`);

const textResource = (uri: URL, resourceId: number) => ({
  uri: uri.toString(),
  mimeType: "text/plain",
  text: `Resource ${resourceId}: This is a plaintext resource created at ${new Date().toLocaleTimeString()}`,
});

const blobResource = (uri: URL, resourceId: number) => ({
  uri: uri.toString(),
  mimeType: "text/plain",
  blob: Buffer.from(`Resource ${resourceId}: This is a base64 blob created at ${new Date().toLocaleTimeString()}`).toString("base64"),
});

const sessionResources = new Map<string, { type: "text" | "blob"; payload: string; resource: Resource }>();

// --- Research Task Logic ---
const STAGES = ["Gathering sources", "Analyzing content", "Synthesizing findings", "Generating report"];
const STAGE_DURATION = 1000;
interface ResearchState {
  topic: string;
  ambiguous: boolean;
  currentStage: number;
  clarification?: string;
  completed: boolean;
  result?: CallToolResult;
}
const researchStates = new Map<string, ResearchState>();

function getInterpretationsForTopic(topic: string): Array<{ const: string; title: string }> {
  const lowerTopic = topic.toLowerCase();
  if (lowerTopic.includes("python")) {
    return [
      { const: "programming", title: "Python programming language" },
      { const: "snake", title: "Python snake species" },
      { const: "comedy", title: "Monty Python comedy group" },
    ];
  }
  return [
    { const: "technical", title: "Technical/scientific perspective" },
    { const: "historical", title: "Historical perspective" },
    { const: "current", title: "Current events/news perspective" },
  ];
}

function generateResearchReport(state: ResearchState): CallToolResult {
  const topic = state.clarification ? `${state.topic} (${state.clarification})` : state.topic;
  const report = `# Research Report: ${topic}\n\n## Research Parameters\n- **Topic**: ${state.topic}\n${state.clarification ? `- **Clarification**: ${state.clarification}` : ""}\n\n## Synthesis\nThis research query was processed through ${STAGES.length} stages:\n${STAGES.map((s, i) => `- Stage ${i + 1}: ${s} ✓`).join("\n")}\n\n---\n\n## About This Demo (SEP-1686: Tasks)\n\nThis tool demonstrates MCP's task-based execution pattern for long-running operations.\n`;
  return { content: [{ type: "text", text: report }] };
}

async function runResearchProcess(taskId: string, args: any, taskStore: any, sendRequest: any): Promise<void> {
  const state = researchStates.get(taskId);
  if (!state) return;
  for (let i = state.currentStage; i < STAGES.length; i++) {
    state.currentStage = i;
    if (state.completed) return;
    await taskStore.updateTaskStatus(taskId, "working", `${STAGES[i]}...`);
    if (i === 2 && state.ambiguous && !state.clarification) {
      await taskStore.updateTaskStatus(taskId, "input_required", `Found multiple interpretations for "${state.topic}". Requesting clarification...`);
      try {
        const elicitResult: ElicitResult = await sendRequest({
          method: "elicitation/create",
          params: {
            message: `The research query "${state.topic}" could have multiple interpretations. Please clarify what you're looking for:`,
            requestedSchema: {
              type: "object",
              properties: {
                interpretation: {
                  type: "string",
                  title: "Clarification",
                  description: "Which interpretation of the topic do you mean?",
                  oneOf: getInterpretationsForTopic(state.topic),
                },
              },
              required: ["interpretation"],
            },
          },
        }, ElicitResultSchema);
        if (elicitResult.action === "accept" && elicitResult.content) {
          state.clarification = (elicitResult.content as any).interpretation || "User accepted without selection";
        } else {
          state.clarification = "User declined/cancelled - using default interpretation";
        }
      } catch (error) {
        state.clarification = "technical (default - elicitation unavailable)";
      }
      await taskStore.updateTaskStatus(taskId, "working", `Continuing with interpretation: "${state.clarification}"...`);
    }
    await new Promise((resolve) => setTimeout(resolve, STAGE_DURATION));
  }
  state.completed = true;
  const result = generateResearchReport(state);
  state.result = result;
  await taskStore.storeTaskResult(taskId, "completed", result);
}

function getSessionResourceURI(name: string): string {
  return `demo://resource/session/${name}`;
}

function registerSessionResource(server: McpServer, resource: Resource, type: "text" | "blob", payload: string): ResourceLink {
  sessionResources.set(resource.uri, { type, payload, resource });
  server.registerResource(resource.name, resource.uri, { mimeType: resource.mimeType, description: resource.description }, async (uri) => {
    const entry = sessionResources.get(uri.toString());
    if (!entry) throw new Error(`Resource not found: ${uri}`);
    return {
      contents: [
        entry.type === "text"
          ? { uri: uri.toString(), mimeType: entry.resource.mimeType, text: entry.payload }
          : { uri: uri.toString(), mimeType: entry.resource.mimeType, blob: entry.payload },
      ],
    };
  });
  return { type: "resource_link", ...resource };
}

// --- Everything Tools ---
server.registerTool("echo", { inputSchema: z.object({ message: z.string() }) }, async (args: any) => {
  return { content: [{ type: "text", text: `Echo: ${args.message}` }] };
});

server.registerTool("get_env", { inputSchema: z.object({}) }, async () => {
  return {
    content: [{ type: "text", text: JSON.stringify(process.env, null, 2) }]
  };
});

server.registerTool("get_sum", {
  inputSchema: z.object({
    a: z.number().describe("First number"),
    b: z.number().describe("Second number")
  })
}, async (args: any) => {
  const sum = args.a + args.b;
  return {
    content: [{ type: "text", text: `The sum of ${args.a} and ${args.b} is ${sum}.` }]
  };
});

server.registerTool("get_structured_content", {
  inputSchema: z.object({
    location: z.enum(["New York", "Chicago", "Los Angeles"]).describe("Choose city")
  })
}, async (args: any) => {
  let weather;
  switch (args.location) {
    case "New York": weather = { temperature: 33, conditions: "Cloudy", humidity: 82 }; break;
    case "Chicago": weather = { temperature: 36, conditions: "Light rain / drizzle", humidity: 82 }; break;
    case "Los Angeles": weather = { temperature: 73, conditions: "Sunny / Clear", humidity: 48 }; break;
  }
  return {
    content: [{ type: "text", text: JSON.stringify(weather) }],
    structuredContent: weather
  };
});

server.registerTool("get_resource_links", {
  inputSchema: z.object({
    count: z.number().min(1).max(10).default(3).describe("Number of resource links to return (1-10)"),
  })
}, async (args: any) => {
  const { count } = args;
  const content: any[] = [{ type: "text", text: `Here are ${count} resource links to resources available in this server:` }];
  for (let i = 1; i <= count; i++) {
    const isOdd = i % 2 !== 0;
    const uri = isOdd ? textResourceUri(i) : blobResourceUri(i);
    const res = isOdd ? textResource(uri, i) : blobResource(uri, i);
    content.push({
      type: "resource_link",
      uri: res.uri,
      name: `${isOdd ? "Text" : "Blob"} Resource ${i}`,
      description: `Resource ${i}: ${isOdd ? "plaintext resource" : "binary blob resource"}`,
      mimeType: res.mimeType,
    });
  }
  return { content };
});

server.registerTool("get_resource_reference", {
  inputSchema: z.object({
    resourceType: z.enum([RESOURCE_TYPE_TEXT, RESOURCE_TYPE_BLOB]).default(RESOURCE_TYPE_TEXT),
    resourceId: z.number().default(1).describe("ID of the resource to fetch"),
  })
}, async (args: any) => {
  const { resourceType, resourceId } = args;
  const uri = resourceType === RESOURCE_TYPE_TEXT ? textResourceUri(resourceId) : blobResourceUri(resourceId);
  const res = resourceType === RESOURCE_TYPE_TEXT ? textResource(uri, resourceId) : blobResource(uri, resourceId);
  return {
    content: [
      { type: "text", text: `Returning resource reference for Resource ${resourceId}:` },
      { type: "resource", resource: res },
      { type: "text", text: `You can access this resource using the URI: ${res.uri}` },
    ],
  };
});

server.registerTool("gzip_file_as_resource", {
  inputSchema: z.object({
    name: z.string().describe("Name of the output file").default("README.md.gz"),
    data: z.string().url().describe("URL or data URI of the file content to compress").default("https://raw.githubusercontent.com/modelcontextprotocol/servers/refs/heads/main/README.md"),
    outputType: z.enum(["resourceLink", "resource"]).default("resourceLink"),
  })
}, async (args: any) => {
  const { name, data, outputType } = args;
  const response = await fetch(data);
  const buffer = await response.arrayBuffer();
  const compressed = gzipSync(Buffer.from(buffer));
  const uri = getSessionResourceURI(name);
  const blob = compressed.toString("base64");
  const mimeType = "application/gzip";
  const resource = { uri, name, mimeType };
  const resourceLink = registerSessionResource(server, resource, "blob", blob);
  if (outputType === "resource") {
    return { content: [{ type: "resource", resource: { uri, mimeType, blob } }] };
  }
  return { content: [resourceLink] };
});

server.registerTool("toggle_subscriber_updates", { inputSchema: z.object({}) }, async (_args, extra) => {
  const sessionId = extra?.sessionId;
  if (subscriberClients.has(sessionId)) {
    const interval = subscriberIntervals.get(sessionId);
    if (interval) clearInterval(interval);
    subscriberIntervals.delete(sessionId);
    subscriberClients.delete(sessionId);
    return { content: [{ type: "text", text: `Stopped simulated resource updates for session ${sessionId}` }] };
  } else {
    subscriberClients.add(sessionId);
    const sendUpdate = async () => {
      // In a real implementation, we would send notifications for subscribed resources
      // For this consolidated server, we'll just log that we would send updates
      console.error(`Simulated resource update for session ${sessionId}`);
    };
    sendUpdate();
    subscriberIntervals.set(sessionId, setInterval(sendUpdate, 5000));
    return { content: [{ type: "text", text: `Started simulated resource updates for session ${sessionId}` }] };
  }
});

server.registerTool("trigger_elicitation_request", { inputSchema: z.object({}) }, async (_args, extra) => {
  const result = await extra.sendRequest({
    method: "elicitation/create",
    params: {
      message: "Please provide your name:",
      requestedSchema: {
        type: "object",
        properties: { name: { type: "string", title: "Name" } },
        required: ["name"],
      },
    },
  }, ElicitResultSchema);
  return { content: [{ type: "text", text: `Elicitation result: ${JSON.stringify(result, null, 2)}` }] };
});

server.registerTool("trigger_sampling_request", {
  inputSchema: z.object({
    prompt: z.string(),
    maxTokens: z.number().default(100),
  })
}, async (args, extra) => {
  const result = await extra.sendRequest({
    method: "sampling/createMessage",
    params: {
      messages: [{ role: "user", content: { type: "text", text: args.prompt } }],
      maxTokens: args.maxTokens,
    },
  }, CreateMessageResultSchema);
  return { content: [{ type: "text", text: `Sampling result: ${JSON.stringify(result, null, 2)}` }] };
});

server.experimental.tasks.registerToolTask("simulate-research-query", {
  title: "Simulate Research Query",
  description: "Simulates a deep research operation.",
  inputSchema: z.object({
    topic: z.string(),
    ambiguous: z.boolean().default(false),
  }),
  execution: { taskSupport: "required" },
}, {
  createTask: async (args: any, extra) => {
    const task = await extra.taskStore.createTask({ ttl: 300000, pollInterval: 1000 });
    const state: ResearchState = { topic: args.topic, ambiguous: args.ambiguous, currentStage: 0, completed: false };
    researchStates.set(task.taskId, state);
    runResearchProcess(task.taskId, args, extra.taskStore, extra.sendRequest).catch(console.error);
    return { task };
  },
  getTask: async (_args, extra) => extra.taskStore.getTask(extra.taskId),
  getTaskResult: async (_args, extra) => {
    const result = await extra.taskStore.getTaskResult(extra.taskId);
    researchStates.delete(extra.taskId);
    return result as CallToolResult;
  },
});

// --- Everything Prompts ---
server.registerPrompt("simple-prompt", {
  title: "Simple Prompt",
  description: "A prompt with no arguments",
}, () => ({
  messages: [{
    role: "user",
    content: { type: "text", text: "This is a simple prompt without arguments." },
  }],
}));

server.registerPrompt("args-prompt", {
  title: "Arguments Prompt",
  description: "A prompt with two arguments, one required and one optional",
  argsSchema: {
    city: z.string().describe("Name of the city"),
    state: z.string().describe("Name of the state").optional(),
  },
}, (args) => {
  const location = `${args?.city}${args?.state ? `, ${args?.state}` : ""}`;
  return {
    messages: [{
      role: "user",
      content: { type: "text", text: `What's weather in ${location}?` },
    }],
  };
});

server.registerPrompt("completable-prompt", {
  title: "Team Management",
  description: "First argument choice narrows values for second argument.",
  argsSchema: {
    department: completable(z.string().describe("Choose the department."), (value) => {
      return ["Engineering", "Sales", "Marketing", "Support"].filter((d) => d.startsWith(value));
    }),
    name: completable(z.string().describe("Choose a team member to lead the selected department."), (value, context) => {
      const department = context?.arguments?.["department"];
      if (department === "Engineering") return ["Alice", "Bob", "Charlie"].filter((n) => n.startsWith(value));
      if (department === "Sales") return ["David", "Eve", "Frank"].filter((n) => n.startsWith(value));
      if (department === "Marketing") return ["Grace", "Henry", "Iris"].filter((n) => n.startsWith(value));
      if (department === "Support") return ["John", "Kim", "Lee"].filter((n) => n.startsWith(value));
      return [];
    }),
  },
}, ({ department, name }) => ({
  messages: [{
    role: "user",
    content: { type: "text", text: `Please promote ${name} to the head of the ${department} team.` },
  }],
}));

server.registerPrompt("resource-prompt", {
  title: "Resource Prompt",
  description: "A prompt that includes an embedded resource reference",
  argsSchema: {
    resourceType: completable(z.string().describe("Type of resource to fetch"), (value) => {
      return RESOURCE_TYPES.filter((t) => t.startsWith(value));
    }),
    resourceId: completable(z.string().describe("ID of the text resource to fetch"), (value) => {
      const id = Number(value);
      return Number.isInteger(id) && id > 0 ? [value] : [];
    }),
  },
}, (args) => {
  const resourceType = args.resourceType as typeof RESOURCE_TYPE_TEXT | typeof RESOURCE_TYPE_BLOB;
  const resourceId = Number(args.resourceId);
  if (!RESOURCE_TYPES.includes(resourceType)) throw new Error(`Invalid resourceType: ${resourceType}`);
  if (!Number.isFinite(resourceId) || !Number.isInteger(resourceId) || resourceId < 1) throw new Error(`Invalid resourceId: ${resourceId}`);
  const uri = resourceType === RESOURCE_TYPE_TEXT ? textResourceUri(resourceId) : blobResourceUri(resourceId);
  const res = resourceType === RESOURCE_TYPE_TEXT ? textResource(uri, resourceId) : blobResource(uri, resourceId);
  return {
    messages: [
      { role: "user", content: { type: "text", text: `This prompt includes the ${resourceType} resource with id: ${resourceId}. Please analyze the following resource:` } },
      { role: "user", content: { type: "resource", resource: res } },
    ],
  };
});

const MCP_TINY_IMAGE = "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";

server.registerTool("get_annotated_message", {
  inputSchema: z.object({
    messageType: z.enum(["error", "success", "debug"]).describe("Type of message"),
    includeImage: z.boolean().default(false).describe("Whether to include an example image")
  })
}, async (args: any) => {
  const content: any[] = [];
  if (args.messageType === "error") {
    content.push({ type: "text", text: "Error: Operation failed", annotations: { priority: 1.0, audience: ["user", "assistant"] } });
  } else if (args.messageType === "success") {
    content.push({ type: "text", text: "Operation completed successfully", annotations: { priority: 0.7, audience: ["user"] } });
  } else if (args.messageType === "debug") {
    content.push({ type: "text", text: "Debug: Cache hit ratio 0.95, latency 150ms", annotations: { priority: 0.3, audience: ["assistant"] } });
  }
  if (args.includeImage) {
    content.push({ type: "image", data: MCP_TINY_IMAGE, mimeType: "image/png", annotations: { priority: 0.5, audience: ["user"] } });
  }
  return { content };
});

server.registerTool("get_tiny_image", { inputSchema: z.object({}) }, async () => {
  return {
    content: [
      { type: "text", text: "Here's the image you requested:" },
      { type: "image", data: MCP_TINY_IMAGE, mimeType: "image/png" },
      { type: "text", text: "The image above is the MCP logo." }
    ]
  };
});

server.registerTool("trigger_long_running_operation", {
  inputSchema: z.object({
    duration: z.number().default(10).describe("Duration of the operation in seconds"),
    steps: z.number().default(5).describe("Number of steps in the operation")
  })
}, async (args: any, extra: any) => {
  const { duration, steps } = args;
  const stepDuration = duration / steps;
  const progressToken = extra._meta?.progressToken;
  for (let i = 1; i < steps + 1; i++) {
    await new Promise((resolve) => setTimeout(resolve, stepDuration * 1000));
    if (progressToken !== undefined) {
      await server.server.notification({
        method: "notifications/progress",
        params: { progress: i, total: steps, progressToken }
      }, { relatedRequestId: extra.requestId });
    }
  }
  return {
    content: [{ type: "text", text: `Long running operation completed. Duration: ${duration} seconds, Steps: ${steps}.` }]
  };
});

server.registerTool("toggle_simulated_logging", { inputSchema: z.object({}) }, async (_args, extra) => {
  const sessionId = extra?.sessionId;
  if (loggingClients.has(sessionId)) {
    const interval = logsUpdateIntervals.get(sessionId);
    if (interval) clearInterval(interval);
    logsUpdateIntervals.delete(sessionId);
    loggingClients.delete(sessionId);
    return { content: [{ type: "text", text: `Stopped simulated logging for session ${sessionId}` }] };
  } else {
    loggingClients.add(sessionId);
    const sendLog = async () => {
      const levels: any[] = ["debug", "info", "notice", "warning", "error", "critical", "alert", "emergency"];
      const level = levels[Math.floor(Math.random() * levels.length)];
      await (server as any).sendLoggingMessage({
        level,
        data: `Simulated ${level} message for session ${sessionId}`
      }, sessionId);
    };
    sendLog();
    logsUpdateIntervals.set(sessionId, setInterval(sendLog, 5000));
    return { content: [{ type: "text", text: `Started simulated logging for session ${sessionId}` }] };
  }
});

server.registerTool("get_roots_list", { inputSchema: z.object({}) }, async () => {
  const clientCapabilities = server.server.getClientCapabilities() || {};
  if (!clientCapabilities.roots) {
    return { content: [{ type: "text", text: "Client does not support roots capability." }] };
  }
  try {
    const response = await server.server.listRoots();
    return { content: [{ type: "text", text: JSON.stringify(response, null, 2) }] };
  } catch (error) {
    return { content: [{ type: "text", text: `Error listing roots: ${error}` }] };
  }
});

// --- Resources ---
server.registerResource("example_resource", "memo://example.txt", {
  mimeType: "text/plain",
  description: "An example text resource"
}, async (uri) => {
  return {
    contents: [{
      uri: uri.toString(),
      mimeType: "text/plain",
      text: "This is an example resource content from the consolidated server."
    }]
  };
});

server.registerResource(
  "Dynamic Text Resource",
  new ResourceTemplate("demo://resource/dynamic/text/{resourceId}", {
    list: undefined,
    complete: {
      resourceId: (value: string) => {
        const id = Number(value);
        return Number.isInteger(id) && id > 0 ? [value] : [];
      }
    }
  }),
  {
    mimeType: "text/plain",
    description: "Plaintext dynamic resource fabricated from the {resourceId} variable.",
  },
  async (uri, variables) => {
    const resourceId = Number(variables.resourceId);
    return {
      contents: [textResource(new URL(uri.toString()), resourceId)],
    };
  }
);

server.registerResource(
  "Dynamic Blob Resource",
  new ResourceTemplate("demo://resource/dynamic/blob/{resourceId}", {
    list: undefined,
    complete: {
      resourceId: (value: string) => {
        const id = Number(value);
        return Number.isInteger(id) && id > 0 ? [value] : [];
      }
    }
  }),
  {
    mimeType: "application/octet-stream",
    description: "Binary dynamic resource fabricated from the {resourceId} variable.",
  },
  async (uri, variables) => {
    const resourceId = Number(variables.resourceId);
    return {
      contents: [blobResource(new URL(uri.toString()), resourceId)],
    };
  }
);

server.registerTool("list_resources", { inputSchema: z.object({}) }, async () => {
  // @ts-ignore - listResources exists on the internal server but might not be in types
  const resources = await (server as any).server.listResources();
  return { content: [{ type: "text", text: JSON.stringify(resources, null, 2) }] };
});

// --- Roots Protocol Support ---
async function updateAllowedDirectoriesFromRoots(requestedRoots: Root[]) {
  const validatedRootDirs = await Promise.all(requestedRoots.map(async (root) => {
    const rootPath = fileURLToPath(root.uri);
    const expanded = expandHome(rootPath);
    const absolute = path.resolve(expanded);
    try {
      const resolved = await fs.realpath(absolute);
      return normalizePath(resolved);
    } catch {
      return normalizePath(absolute);
    }
  }));
  if (validatedRootDirs.length > 0) {
    allowedDirectories = [...validatedRootDirs];
    console.error(`Updated allowed directories from MCP roots: ${validatedRootDirs.length} valid directories`);
  }
}

server.server.setNotificationHandler(RootsListChangedNotificationSchema, async () => {
  try {
    const response = await server.server.listRoots();
    if (response && 'roots' in response) {
      await updateAllowedDirectoriesFromRoots(response.roots);
    }
  } catch (error) {
    console.error("Failed to request roots from client:", error);
  }
});

server.server.oninitialized = async () => {
  const clientCapabilities = server.server.getClientCapabilities();
  if (clientCapabilities?.roots) {
    try {
      const response = await server.server.listRoots();
      if (response && 'roots' in response) {
        await updateAllowedDirectoriesFromRoots(response.roots);
      }
    } catch (error) {
      console.error("Failed to request initial roots from client:", error);
    }
  }
};

async function main() {
  const args = process.argv.slice(2);
  const transportType = args[0] === "sse" ? "sse" : "stdio";
  const initialDirs = transportType === "sse" ? args.slice(1) : args;

  const initialAllowedDirs = await Promise.all(initialDirs.map(async (dir) => {
    const expanded = expandHome(dir);
    const absolute = path.resolve(expanded);
    try {
      const resolved = await fs.realpath(absolute);
      return normalizePath(resolved);
    } catch {
      return normalizePath(absolute);
    }
  }));
  setAllowedDirectories(initialAllowedDirs);

  // Register static file resources from docs
  const docsDir = path.join(path.dirname(fileURLToPath(import.meta.url)), '..', 'everything', 'docs');
  try {
    const entries = readdirSync(docsDir);
    for (const name of entries) {
      const fullPath = path.join(docsDir, name);
      const st = statSync(fullPath);
      if (!st.isFile()) continue;
      const uri = `demo://resource/static/document/${encodeURIComponent(name)}`;
      const mimeType = name.endsWith(".md") ? "text/markdown" : name.endsWith(".json") ? "application/json" : "text/plain";
      server.registerResource(name, uri, { mimeType, description: `Static document: ${name}` }, async (uri) => {
        return { contents: [{ uri: uri.toString(), mimeType, text: readFileSync(fullPath, "utf-8") }] };
      });
    }
  } catch (e) {
    console.error("Failed to register static resources:", e);
  }

  if (transportType === "stdio") {
    const transport = new StdioServerTransport();
    await server.connect(transport);
    console.error("Consolidated MCP Server running on stdio");
  } else {
    const app = express();
    app.use(cors());
    const transports = new Map<string, SSEServerTransport>();

    app.get("/sse", async (_req, res) => {
      const transport = new SSEServerTransport("/message", res);
      transports.set(transport.sessionId, transport);
      await server.connect(transport);
      console.error(`Client connected: ${transport.sessionId}`);
      server.server.onclose = async () => {
        transports.delete(transport.sessionId);
        console.error(`Client disconnected: ${transport.sessionId}`);
      };
    });

    app.post("/message", async (req, res) => {
      const sessionId = req.query.sessionId as string;
      const transport = transports.get(sessionId);
      if (transport) await transport.handlePostMessage(req, res);
      else res.status(404).send("Session not found");
    });

    const PORT = process.env.PORT || 3001;
    app.listen(PORT, () => {
      console.error(`Consolidated MCP Server running on SSE at http://localhost:${PORT}/sse`);
    });
  }

  // Cleanup on exit
  const cleanup = () => {
    for (const interval of logsUpdateIntervals.values()) clearInterval(interval);
    for (const interval of subscriberIntervals.values()) clearInterval(interval);
    taskStore.cleanup();
  };

  process.on('SIGINT', cleanup);
  process.on('SIGTERM', cleanup);
}

main().catch((error) => {
  console.error("Fatal error:", error);
  process.exit(1);
});
