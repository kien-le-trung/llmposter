import type { RoundPlayer } from "@/lib/api";

const FIRST_NAMES = [
  "Maya", "Alex", "Jordan", "Priya", "Leo", "Sofia", "Noah", "Amara",
  "Ethan", "Isabel", "Omar", "Chloe", "Felix", "Nina", "Luca", "Zoe",
  "Arjun", "Elena", "Theo", "Aisha", "Sam", "Clara", "Miles", "Yuna",
  "Jamie", "Leila", "Oscar", "Iris", "Ravi", "Freya", "Daniel", "Naomi",
];
const LAST_NAMES = ["Brooks", "Chen", "Patel", "Rivera", "Kim", "Reed", "Morgan", "Silva"];

export function assignPlayerNames(players: RoundPlayer[]): Record<string, string> {
  const pool = [...FIRST_NAMES];
  for (let index = pool.length - 1; index > 0; index -= 1) {
    const other = Math.floor(Math.random() * (index + 1));
    [pool[index], pool[other]] = [pool[other], pool[index]];
  }
  let next = 0;
  return Object.fromEntries(players.map((player) => {
    if (player.kind === "human") return [player.id, "You"];
    const index = next++;
    // Larger tables get distinct full names after the first-name pool is exhausted.
    const name = index < pool.length ? pool[index]
      : `${pool[index % pool.length]} ${LAST_NAMES[Math.floor(index / pool.length - 1) % LAST_NAMES.length]}`;
    return [player.id, name];
  }));
}

export function playerDisplayName(
  players: RoundPlayer[], aliases: Record<string, string>, id: string | null, fallback: string,
): string {
  if (id && aliases[id]) return aliases[id];
  const normalized = fallback.trim().toLowerCase();
  const player = players.find((candidate) =>
    candidate.id.toLowerCase() === normalized || candidate.name.toLowerCase() === normalized,
  );
  if (player) return aliases[player.id] ?? player.name;
  if (["you", "human"].includes(normalized)) return "You";
  return fallback;
}
