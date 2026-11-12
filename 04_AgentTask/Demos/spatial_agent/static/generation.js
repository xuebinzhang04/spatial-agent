// Task generation section: controls -> local state / global memory variables -> expectation checks.
(function () {
  const data = window.SPATIAL_GENERATION;
  const select = document.getElementById("generation-task");
  const body = document.getElementById("generation-body");
  if (!data || !select || !body) return;
  const model = data.variable_model;
  const escape = s => String(s).replace(/[&<>"]/g, c => ({"&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;"}[c]));
  const label = s => s.replace(/_/g, " ");
  const SOURCE = {live: "fresh generation", recorded: "recorded episode"};

  function show(v) {
    if (Array.isArray(v)) return v.map(show).join(" · ");
    if (v === true) return "yes";
    if (v === false) return "no";
    if (v === null || v === undefined) return "—";
    if (typeof v === "object") return JSON.stringify(v);
    return String(v);
  }

  function overrides(controls) {
    const config = controls.task_config || {};
    return Object.keys(config).filter(k => k !== "split" && k !== "image_size")
      .map(k => k + "=" + show(config[k])).join(", ");
  }

  function checkCell(item) {
    if (!item.admitted) return '<span class="gen-fail">rejected</span><br><small>' + escape(item.reset_error || "") + "</small>";
    const count = s => item.checks.filter(c => c.status === s).length;
    const issues = item.checks.filter(c => c.status !== "pass")
      .map(c => '<li class="gen-' + c.status + '">' + escape(c.check) + (c.detail ? " — " + escape(c.detail) : "") + "</li>").join("");
    const head = '<span class="gen-pass">✓ ' + count("pass") + "</span> " +
      (count("warn") ? '<span class="gen-warn">⚠ ' + count("warn") + "</span> " : "") +
      (count("fail") ? '<span class="gen-fail">✗ ' + count("fail") + "</span>" : "");
    return issues ? "<details><summary>" + head + "</summary><ul>" + issues + "</ul></details>" : head;
  }

  function render(task) {
    const spec = model.tasks[task];
    const names = Object.keys(spec.global_memory);
    const local = Object.keys(model.shared.local_state).concat(spec.local_state_extra);
    const rows = [].concat.apply([], data.audits.map(audit => audit.cases.filter(c => c.task === task)));
    const summaries = data.audits.filter(a => a.summary[task]).map(a => {
      const s = a.summary[task];
      const job = a.slurm_job_id ? " · Slurm " + a.slurm_job_id : "";
      return "<li><strong>" + escape(SOURCE[a.source] || a.source) + "</strong> (" + escape(a.run_id + job) + "): " +
        s.admitted + "/" + s.cases + " admitted, " + s.distinct_variable_sets + " distinct global-memory states" +
        (s.distinct_start_positions ? ", " + s.distinct_start_positions + " distinct start positions" : "") +
        (s.failed.length ? ', <span class="gen-fail">' + s.failed.length + " failed checks</span>" : "") +
        (s.warned.length ? ', <span class="gen-warn">' + s.warned.length + " warnings</span>" : "") + "</li>";
    }).join("");
    const balance = task === "shape_identification" && data.audits.map(a => a.shape_answer_balance).find(Boolean);
    body.innerHTML =
      '<div class="gen-grid">' +
        "<div><h4>Global memory lifecycle</h4><dl class=\"task-details\">" +
          ["build", "update", "freeze"].map(k => "<dt>" + k + "</dt><dd>" + escape(spec.lifecycle[k]) + "</dd>").join("") +
          "<dt>scored by</dt><dd>" + escape(spec.memory_reward) + "</dd></dl></div>" +
        "<div><h4>Local state (public, per step)</h4><p>" + local.map(k => "<code>" + escape(k) + "</code>").join(" ") + "</p>" +
          "<h4>Controls</h4><table class=\"table is-narrow\"><tbody>" +
          Object.keys(spec.controls).map(k => "<tr><td><code>" + escape(k) + "</code></td><td>" + escape(spec.controls[k]) + "</td></tr>").join("") +
          "</tbody></table></div>" +
      "</div>" +
      "<h4>Generated cases</h4><ul class=\"gen-summary\">" + summaries + "</ul>" +
      (balance ? "<p class=\"caption\">Answer position over " + balance.seeds + " seeds (L" + balance.difficulty + "): " +
        escape(Object.keys(balance.counts).map(k => k.replace("candidate_", "#") + " " + balance.counts[k]).join(", ")) +
        "; χ² = " + balance.chi_square + " (df = 4; above 13.3 would indicate positional bias).</p>" : "") +
      '<div class="table-container"><table class="table is-fullwidth is-narrow gen-table"><thead><tr><th>Case</th><th>Source</th><th>L</th><th>Seed</th><th>Overrides</th>' +
      names.map(n => '<th title="' + escape(spec.global_memory[n]) + '">' + escape(label(n)) + "</th>").join("") +
      "<th>Checks</th></tr></thead><tbody>" +
      rows.map(r => "<tr><td>" + escape(r.case) + "</td><td>" + escape(SOURCE[r.source] || r.source) + "</td><td>" +
        r.controls.difficulty + "</td><td>" + r.controls.seed + "</td><td>" + escape(overrides(r.controls)) + "</td>" +
        names.map(n => "<td>" + escape(r.global_memory ? show(r.global_memory[n]) : "—") + "</td>").join("") +
        "<td>" + checkCell(r) + "</td></tr>").join("") +
      "</tbody></table></div>";
  }

  select.innerHTML = Object.keys(model.tasks).map(t => '<option value="' + t + '">' + escape(label(t)) + "</option>").join("");
  select.value = Object.keys(model.tasks)[0];
  select.addEventListener("change", () => render(select.value));
  render(select.value);
})();
