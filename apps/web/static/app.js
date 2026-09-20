/* Bare-bones, no dependencies. Same-origin fetch only. */
(function () {
  "use strict";
  var $ = function (id) { return document.getElementById(id); };
  var pairs = [
    ["num_colors", "o_colors"], ["max_dim", "o_dim"], ["fps", "o_fps"],
    ["merge_area", "o_merge"], ["keyframe_step", "o_kf"],
    ["scene_threshold", "o_thr"], ["scene_min_len", "o_slen"]
  ];
  pairs.forEach(function (p) {
    var el = $(p[0]), out = $(p[1]);
    var sync = function () { out.textContent = el.value; };
    el.addEventListener("input", sync); sync();
  });

  var status = $("status"), go = $("go"), result = $("result");
  var fileInput = $("video"), uploaded = $("uploaded");
  var uploadedUrl = null;
  var timer = null;

  function setStatus(msg) { status.textContent = msg; }

  function poll(id) {
    fetch("/api/status?id=" + encodeURIComponent(id), { credentials: "same-origin" })
      .then(function (r) { return r.json().then(function (j) { return { ok: r.ok, j: j }; }); })
      .then(function (res) {
        if (!res.ok) { setStatus("Error: " + (res.j.error || res.ok)); stop(); return; }
        var j = res.j;
        if (j.status === "done") {
          stop();
          setStatus("Done in " + (j.report.elapsed_s || "?") + "s.");
          $("preview").src = "/preview?id=" + encodeURIComponent(id);
          $("dl_json").href = "/api/download?id=" + encodeURIComponent(id) + "&kind=json";
          $("dl_mp4").href = "/api/download?id=" + encodeURIComponent(id) + "&kind=mp4";
          $("report").textContent =
            (j.report.num_frames || "?") + " frames, " +
            (j.report.num_tracks || "?") + " tracks, " +
            (j.report.size_kb || "?") + " KB, mode=" + (j.report.mode || "?");
          result.classList.remove("hidden");
        } else if (j.status === "error") {
          stop();
          setStatus("Conversion failed: " + (j.error || "unknown"));
        } else {
          setStatus("Working… (" + j.status + ")");
        }
      })
      .catch(function () { setStatus("Network error while polling."); stop(); });
  }
  function stop() { if (timer) { clearInterval(timer); timer = null; } go.disabled = false; }

  // Local preview of the uploaded file (never touches the server).
  fileInput.addEventListener("change", function () {
    var f = fileInput.files[0];
    if (uploadedUrl) { URL.revokeObjectURL(uploadedUrl); uploadedUrl = null; }
    uploaded.removeAttribute("src");
    if (!f) return;
    if (f.size > 50 * 1024 * 1024) { setStatus("File too big (50MB max)."); return; }
    uploadedUrl = URL.createObjectURL(f);
    uploaded.src = uploadedUrl;
    setStatus("");
    var hint = document.getElementById("upload_hint");
    if (hint) hint.textContent = f.name + " (" + Math.round(f.size / 1024) + " KB)";
  });

  go.addEventListener("click", function () {
    var f = fileInput.files[0];
    if (!f) { setStatus("Pick a video file first."); return; }
    if (f.size > 50 * 1024 * 1024) { setStatus("File too big (50MB max)."); return; }
    result.classList.add("hidden");
    $("preview").removeAttribute("src");
    go.disabled = true;
    setStatus("Uploading…");
    var fd = new FormData();
    fd.append("video", f, f.name);
    ["mode", "flow", "num_colors", "max_dim", "fps", "merge_area",
     "keyframe_step", "scene_threshold", "scene_min_len"].forEach(function (k) {
      fd.append(k, $(k).value);
    });
    fetch("/api/jobs", { method: "POST", body: fd, credentials: "same-origin" })
      .then(function (r) { return r.json().then(function (j) { return { ok: r.ok, j: j }; }); })
      .then(function (res) {
        if (!res.ok || !res.j.id) { setStatus("Error: " + (res.j.error || "upload failed")); go.disabled = false; return; }
        setStatus("Queued…");
        timer = setInterval(function () { poll(res.j.id); }, 2000);
        poll(res.j.id);
      })
      .catch(function () { setStatus("Upload failed (network)."); go.disabled = false; });
  });
})();
