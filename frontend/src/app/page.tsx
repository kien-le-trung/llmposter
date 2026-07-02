import { GameConsole } from "@/app/game-console";
import { getAgents } from "@/lib/api";

export default async function Home() {
  const agents = await getAgents().catch(() => []);

  return (
    <main className="page">
      <div className="shell">
        <header className="header">
          <div>
            <p className="eyebrow">Social deduction table</p>
            <h1>Imposter</h1>
          </div>
        </header>

        <GameConsole agents={agents} />
      </div>
    </main>
  );
}
