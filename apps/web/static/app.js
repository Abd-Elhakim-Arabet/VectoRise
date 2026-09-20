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
    var fill = document.getElementById("f_" + p[0]);
    var unit = "";
    var wrap = el.closest ? el.closest(".fslider") : null;
    if (wrap && wrap.getAttribute) unit = wrap.getAttribute("data-unit") || "";
    var sync = function () {
      out.textContent = unit ? el.value + unit : el.value;
      if (fill) {
        var min = parseFloat(el.min), max = parseFloat(el.max),
            v = parseFloat(el.value);
        var pct = ((v - min) / (max - min)) * 100;
        fill.style.width = pct + "%";
      }
    };
    el.addEventListener("input", sync); sync();
  });

  var status = $("status"), go = $("go"), result = $("result");
  var fileInput = $("video"), uploaded = $("uploaded");
  var uploadedUrl = null;
  var timer = null;

  function setStatus(msg, isErr) {
    status.textContent = msg;
    status.classList.toggle("error", !!isErr);
  }

  function poll(id) {
    fetch("/api/status?id=" + encodeURIComponent(id), { credentials: "same-origin" })
      .then(function (r) { return r.json().then(function (j) { return { ok: r.ok, j: j }; }); })
      .then(function (res) {
        if (!res.ok) { setStatus("Error: " + (res.j.error || res.ok), true); pHide(); stop(); return; }
        var j = res.j;
        if (j.status === "done") {
          stop();
          setStatus("Done in " + (j.report.elapsed_s || "?") + "s — fresh vector below.");
          pDone();
          $("preview").src = "/preview?id=" + encodeURIComponent(id);
          document.getElementById("panel_res").classList.add("has-media");
          $("dl_json").href = "/api/download?id=" + encodeURIComponent(id) + "&kind=json";
          $("dl_mp4").href = "/api/download?id=" + encodeURIComponent(id) + "&kind=mp4";
          $("report").textContent =
            (j.report.num_frames || "?") + " frames, " +
            (j.report.num_tracks || "?") + " tracks, " +
            (j.report.size_kb || "?") + " KB, mode=" + (j.report.mode || "?");
        } else if (j.status === "error") {
          stop();
          $("report").textContent = "Conversion failed: " + (j.error || "unknown");
          $("report").classList.add("error");
          setStatus("Conversion failed: " + (j.error || "unknown"), true);
          pHide();
        } else {
          setStatus("Working… (" + j.status + ")");
        }
      })
      .catch(function () { setStatus("Network error while polling.", true); pHide(); stop(); });
  }
  function stop() { if (timer) { clearInterval(timer); timer = null; } go.disabled = false; }

  var pwrap = $("pwrap"), pbar = $("pbar");
  function pFill(pct) {
    pwrap.hidden = false;
    pwrap.classList.remove("busy");
    pbar.style.width = pct + "%";
  }
  function pBusy() {
    pwrap.hidden = false;
    pwrap.classList.add("busy");
  }
  function pDone() {
    pwrap.hidden = false;
    pwrap.classList.remove("busy");
    pbar.style.width = "100%";
  }
  function pHide() {
    pwrap.hidden = true;
    pwrap.classList.remove("busy");
  }

  // Local preview of the uploaded file (never touches the server).
  fileInput.addEventListener("change", function () {
    var f = fileInput.files[0];
    if (uploadedUrl) { URL.revokeObjectURL(uploadedUrl); uploadedUrl = null; }
    uploaded.removeAttribute("src");
    document.getElementById("panel_up").classList.remove("has-media");
    if (!f) return;
    if (f.size > 10 * 1024 * 1024) { setStatus("File too big (10MB · 10s max).", true); return; }
    uploadedUrl = URL.createObjectURL(f);
    uploaded.src = uploadedUrl;
    document.getElementById("panel_up").classList.add("has-media");
    setStatus("");
    var hint = document.getElementById("upload_hint");
    if (hint) hint.textContent = f.name + " (" + Math.round(f.size / 1024) + " KB)";
  });

  go.addEventListener("click", function () {
    var f = fileInput.files[0];
    if (!f) { setStatus("Pick a video file first.", true); return; }
    if (f.size > 10 * 1024 * 1024) { setStatus("File too big (10MB · 10s max).", true); return; }
    $("preview").removeAttribute("src");
    document.getElementById("panel_res").classList.remove("has-media");
    $("report").textContent = "Vectorising…";
    $("report").classList.remove("error");
    go.disabled = true;
    pFill(0);
    setStatus("Uploading…");
    var fd = new FormData();
    fd.append("video", f, f.name);
    ["mode", "flow", "num_colors", "max_dim", "fps", "merge_area",
     "keyframe_step", "scene_threshold", "scene_min_len"].forEach(function (k) {
      fd.append(k, $(k).value);
    });
    // XHR (not fetch) so the upload phase reports real progress.
    var xhr = new XMLHttpRequest();
    xhr.open("POST", "/api/jobs");
    xhr.withCredentials = true;
    xhr.upload.onprogress = function (e) {
      if (e.lengthComputable) pFill(Math.round(e.loaded / e.total * 100));
    };
    xhr.onload = function () {
      var res = {};
      try { res = JSON.parse(xhr.responseText); } catch (err) { res = {}; }
      if (xhr.status < 200 || xhr.status >= 300 || !res.id) {
        setStatus("Error: " + (res.error || "upload failed"), true);
        pHide(); go.disabled = false; return;
      }
      pBusy();
      setStatus("Queued…");
      timer = setInterval(function () { poll(res.id); }, 2000);
      poll(res.id);
    };
    xhr.onerror = function () {
      setStatus("Upload failed (network).", true); pHide(); go.disabled = false;
    };
    xhr.send(fd);
  });
})();
