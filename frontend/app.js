const apiUrlInput = document.getElementById("apiUrl");
const resumeZone = document.getElementById("resumeZone");
const resumeFile = document.getElementById("resumeFile");
const resumeText = document.getElementById("resumeText");
const jobText = document.getElementById("jobText");
const matchBtn = document.getElementById("matchBtn");
const resultSection = document.getElementById("resultSection");
const scoreDisplay = document.getElementById("scoreDisplay");
const explanationsEl = document.getElementById("explanations");
const resultError = document.getElementById("resultError");

let resumeFileData = null;

function baseUrl() {
  return apiUrlInput.value.replace(/\/$/, "");
}

function setLoading(loading) {
  matchBtn.disabled = loading;
  matchBtn.textContent = loading ? "Matching…" : "Match";
}

function showError(msg) {
  resultSection.hidden = false;
  scoreDisplay.textContent = "";
  explanationsEl.innerHTML = "";
  resultError.textContent = msg;
  resultError.hidden = false;
}

function showResult(score, explanations) {
  resultSection.hidden = false;
  resultError.hidden = true;
  scoreDisplay.textContent = `Score: ${score}/100`;
  explanationsEl.innerHTML = explanations.map((e) => `<li>${escapeHtml(e)}</li>`).join("");
}

function escapeHtml(s) {
  const div = document.createElement("div");
  div.textContent = s;
  return div.innerHTML;
}

resumeZone.addEventListener("click", () => resumeFile.click());
resumeZone.addEventListener("dragover", (e) => {
  e.preventDefault();
  resumeZone.classList.add("dragover");
});
resumeZone.addEventListener("dragleave", () => resumeZone.classList.remove("dragover"));
resumeZone.addEventListener("drop", (e) => {
  e.preventDefault();
  resumeZone.classList.remove("dragover");
  const f = e.dataTransfer?.files?.[0];
  if (f && /\.(pdf|docx?)$/i.test(f.name)) {
    resumeFile.files = e.dataTransfer.files;
    resumeZone.querySelector("p").textContent = f.name;
    resumeFileData = f;
  }
});

resumeFile.addEventListener("change", (e) => {
  const f = e.target.files?.[0];
  if (f) {
    resumeZone.querySelector("p").textContent = f.name;
    resumeFileData = f;
  }
});

function hasInput() {
  return (resumeFileData || resumeText.value.trim()) && jobText.value.trim();
}

resumeText.addEventListener("input", () => { matchBtn.disabled = !hasInput(); });
jobText.addEventListener("input", () => { matchBtn.disabled = !hasInput(); });
resumeFile.addEventListener("change", () => { matchBtn.disabled = !hasInput(); });

async function createDocument(type, fileOrText) {
  const form = new FormData();
  form.append("type", type);
  if (typeof fileOrText === "string") {
    form.append("text", fileOrText);
  } else {
    form.append("file", fileOrText);
  }
  const r = await fetch(`${baseUrl()}/api/documents`, {
    method: "POST",
    body: form,
  });
  if (!r.ok) {
    const err = await r.json().catch(() => ({ detail: r.statusText }));
    throw new Error(err.detail || "Failed to create document");
  }
  return r.json();
}

async function extractDocument(id) {
  const r = await fetch(`${baseUrl()}/api/documents/${id}/extract`, { method: "POST" });
  if (!r.ok) {
    const err = await r.json().catch(() => ({ detail: r.statusText }));
    throw new Error(err.detail || "Extraction failed");
  }
  return r.json();
}

async function match(resumeId, jobId) {
  const r = await fetch(`${baseUrl()}/api/match`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ resume_id: resumeId, job_id: jobId }),
  });
  if (!r.ok) {
    const err = await r.json().catch(() => ({ detail: r.statusText }));
    throw new Error(err.detail || "Match failed");
  }
  return r.json();
}

matchBtn.addEventListener("click", async () => {
  if (!hasInput()) return;
  setLoading(true);
  try {
    const resumeInput = resumeFileData || resumeText.value.trim();
    const jobInput = jobText.value.trim();

    const resumeDoc = await createDocument("resume", resumeInput);
    const jobDoc = await createDocument("job_description", jobInput);

    await extractDocument(resumeDoc.id);
    await extractDocument(jobDoc.id);

    const result = await match(resumeDoc.id, jobDoc.id);
    showResult(result.score, result.explanations || []);
  } catch (err) {
    showError(err.message || "Something went wrong. Is the API running at " + baseUrl() + "?");
  } finally {
    setLoading(false);
  }
});
