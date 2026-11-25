// Published evaluations only. Missing entries remain unranked, never 0%.
// See website/README.md for the result format and publication requirements.
window.OWB_LEADERBOARD = {
  worlds: [
    { id: "minecraft", name: "Minecraft", short: "MC" },
    { id: "teyvat", name: "Teyvat", short: "TY" },
    { id: "wildlands", name: "Wildlands", short: "WL" },
    { id: "skylands", name: "Skylands", short: "SK" },
    { id: "mobile", name: "Mobile manipulation", short: "MM" },
  ],
  conditions: [
    { id: "native", name: "Native observations", description: "Only the environment’s policy observations. No privileged simulator state." },
    { id: "perception", name: "Perception oracle", description: "Oracle-assisted perception, evaluated separately from native observations and privileged access." },
    { id: "privileged", name: "Privileged", description: "Privileged simulator information. Results are kept separate from observation-only agents." },
  ],
  models: [
    { id: "claude-opus-5-5", name: "Claude Opus 5.5", organization: "Anthropic", mark: "A", tone: "clay", results: {} },
    { id: "gpt-6-astra", name: "GPT-6-Astra", organization: "OpenAI", mark: "O", tone: "teal", results: {} },
    { id: "gemini-4-argon", name: "Gemini 4 Argon", organization: "Google", mark: "G", tone: "blue", results: {} },
  ],
};
