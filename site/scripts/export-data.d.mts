import type { ResearchData, Example } from "../src/lib/types";
export function sha(path: string): string;
export function validateExample(value: unknown): Example;
export function validateData(value: unknown): ResearchData;
export function buildResearch(root?: string): ResearchData;
export function exportResearch(root?: string): ResearchData;
