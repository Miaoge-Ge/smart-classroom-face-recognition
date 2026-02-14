const video = document.getElementById('videoElement');
const canvas = document.getElementById('canvasElement');
const startBtn = document.getElementById('startBtn');
const stopBtn = document.getElementById('stopBtn');
const countBadge = document.getElementById('count-badge');
const courseSelect = document.getElementById('courseSelect');
const cameraSelect = document.getElementById('cameraSelect');
const wsStatusBadge = document.getElementById('wsStatusBadge');
const fpsBadge = document.getElementById('fpsBadge');
const latencyBadge = document.getElementById('latencyBadge');
const processingBadge = document.getElementById('processingBadge');
const expectedCountEl = document.getElementById('expectedCount');
const actualCountEl = document.getElementById('actualCount');
const absentCountEl = document.getElementById('absentCount');
const absentListEl = document.getElementById('absentList');

if (canvas) {
    const ctx = canvas.getContext('2d');

    let stream = null;
    let ws = null;
    let isRunning = false;
    let lastClientTs = null;
    let recvFrames = 0;
    let lastFpsTick = performance.now();
    let lastLatencyMs = null;
    let lastProcessingMs = null;

    let runtimeSettings = {
        capture: { width: 1280, height: 720, frame_interval_ms: 33, jpeg_quality: 0.7 }
    };

    let wsState = 'DISCONNECTED';
    let sendState = 'IDLE';
    let inFlight = false;
    let inFlightSince = null;
    let schedulerTimer = null;
    let tempCanvas = null;
    let tempCtx = null;
    let activeTaskId = null;

    canvas.width = runtimeSettings.capture.width;
    canvas.height = runtimeSettings.capture.height;

    if (startBtn) startBtn.addEventListener('click', startCamera);
    if (stopBtn) stopBtn.addEventListener('click', stopCamera);
    if (cameraSelect) cameraSelect.addEventListener('change', onCameraChange);
    if (courseSelect) courseSelect.addEventListener('change', () => {
        activeTaskId = null;
        resetStatsUI();
    });

    if (navigator.mediaDevices && navigator.mediaDevices.addEventListener) {
        navigator.mediaDevices.addEventListener('devicechange', () => refreshCameraList(true));
    } else if (navigator.mediaDevices) {
        navigator.mediaDevices.ondevicechange = () => refreshCameraList(true);
    }

    async function loadRuntimeSettings() {
        try {
            const res = await fetch('/api/runtime_settings', { method: 'GET' });
            if (!res.ok) return;
            const data = await res.json();
            if (data && data.capture) runtimeSettings = data;
        } catch (_) { }
    }

    function getQueryParam(name) {
        try {
            const u = new URL(window.location.href);
            return u.searchParams.get(name);
        } catch (_) {
            return null;
        }
    }

    async function bootstrapTaskFromUrl() {
        const tid = getQueryParam('task_id');
        if (!tid) return;
        const n = Number(tid);
        if (!Number.isFinite(n) || n <= 0) return;
        activeTaskId = Math.trunc(n);
        try {
            const res = await fetch('/api/attendance_tasks', { method: 'GET' });
            if (!res.ok) return;
            const data = await res.json();
            const tasks = data && Array.isArray(data.tasks) ? data.tasks : [];
            const t = tasks.find(x => Number(x.task_id) === activeTaskId);
            if (t && courseSelect && t.course_id != null) {
                courseSelect.value = String(t.course_id);
            }
        } catch (_) { }
    }

    function applyCaptureSize() {
        const width = video && video.videoWidth ? video.videoWidth : (runtimeSettings.capture.width || 1280);
        const height = video && video.videoHeight ? video.videoHeight : (runtimeSettings.capture.height || 720);

        const container = document.getElementById('video-container');
        if (container) {
            container.style.width = `${width}px`;
            container.style.height = `${height}px`;
        }
        if (video) {
            video.style.width = `${width}px`;
            video.style.height = `${height}px`;
        }
        if (canvas) {
            canvas.width = width;
            canvas.height = height;
            canvas.style.width = `${width}px`;
            canvas.style.height = `${height}px`;
        }
    }

    function ensureCameraConsent() {
        const key = 'camera_consent_v1';
        if (localStorage.getItem(key) === '1') return true;
        const ok = window.confirm(
            '本模块将申请摄像头权限用于课堂考勤识别与本地预览，并可能将采集画面通过网络发送至后端进行识别处理。是否继续？'
        );
        if (ok) localStorage.setItem(key, '1');
        return ok;
    }

    async function refreshCameraList(preserveSelection) {
        if (!navigator.mediaDevices || !navigator.mediaDevices.enumerateDevices || !cameraSelect) return;
        let selected = cameraSelect.value;
        if (!preserveSelection) selected = '';
        try {
            const devices = await navigator.mediaDevices.enumerateDevices();
            const cams = devices.filter(d => d.kind === 'videoinput');
            const options = [{ value: '', label: '默认摄像头' }];
            cams.forEach((d, idx) => options.push({ value: d.deviceId, label: d.label || `摄像头 ${idx + 1}` }));

            cameraSelect.innerHTML = '';
            options.forEach(o => {
                const opt = document.createElement('option');
                opt.value = o.value;
                opt.textContent = o.label;
                cameraSelect.appendChild(opt);
            });

            if (selected && options.some(o => o.value === selected)) cameraSelect.value = selected;
            else cameraSelect.value = '';
        } catch (_) { }
    }

    function stopStreamTracks() {
        if (!stream) return;
        stream.getTracks().forEach(track => {
            try {
                track.onended = null;
                track.stop();
            } catch (_) { }
        });
        stream = null;
    }

    async function openStream(deviceId) {
        if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
            throw new Error("Browser does not support getUserMedia");
        }

        stopStreamTracks();

        const maxWidth = runtimeSettings.capture.width || 1280;
        const maxHeight = runtimeSettings.capture.height || 720;
        const mkConstraints = (mode) => {
            const videoConstraints = {};
            if (deviceId) videoConstraints.deviceId = { exact: deviceId };

            if (mode === 'prefer_hd') {
                videoConstraints.width = { ideal: maxWidth };
                videoConstraints.height = { ideal: maxHeight };
                videoConstraints.frameRate = { ideal: 30, max: 30 };
            } else if (mode === 'low') {
                videoConstraints.width = { ideal: 640 };
                videoConstraints.height = { ideal: 480 };
                videoConstraints.frameRate = { ideal: 30, max: 30 };
            } else if (mode === 'bare') {
                videoConstraints.frameRate = { ideal: 30, max: 30 };
            }
            return { video: videoConstraints };
        };

        const attempts = ['prefer_hd', 'bare', 'low'];
        let lastErr = null;
        for (const mode of attempts) {
            try {
                stream = await navigator.mediaDevices.getUserMedia(mkConstraints(mode));
                lastErr = null;
                break;
            } catch (e) {
                lastErr = e;
            }
        }
        if (!stream && lastErr) throw lastErr;
        video.srcObject = stream;

        await new Promise((resolve, reject) => {
            const t = setTimeout(() => reject(new Error('Video init timeout')), 10000);
            video.onloadedmetadata = () => {
                clearTimeout(t);
                video.play().then(resolve).catch(resolve);
            };
        });

        try {
            const tracks = stream.getVideoTracks();
            if (tracks && tracks[0]) {
                const s = tracks[0].getSettings ? tracks[0].getSettings() : null;
                if (s && s.width && s.height) {
                    runtimeSettings.capture.width = s.width;
                    runtimeSettings.capture.height = s.height;
                } else if (video.videoWidth && video.videoHeight) {
                    runtimeSettings.capture.width = video.videoWidth;
                    runtimeSettings.capture.height = video.videoHeight;
                }
            }
        } catch (_) { }

        applyCaptureSize();

        const tracks = stream.getVideoTracks();
        if (tracks && tracks[0]) {
            tracks[0].onended = () => {
                alert('摄像头已断开或权限被撤销，已停止采集。');
                stopCamera();
            };
        }
    }

    function connectWebSocket() {
        const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
        wsState = 'CONNECTING';
        ws = new WebSocket(`${protocol}//${window.location.host}/ws/stream`);
        updateWsStatus();

        ws.onopen = () => {
            wsState = 'OPEN';
            isRunning = true;
            if (startBtn) startBtn.disabled = true;
            if (stopBtn) stopBtn.disabled = false;
            recvFrames = 0;
            lastFpsTick = performance.now();
            lastClientTs = null;
            inFlight = false;
            inFlightSince = null;
            startScheduler();
            updateWsStatus();
        };

        ws.onmessage = (event) => {
            const data = JSON.parse(event.data);
            inFlight = false;
            inFlightSince = null;
            recvFrames += 1;
            drawResults(data.faces);
            if (countBadge) countBadge.textContent = data.attendance_count;
            if (typeof data.expected_count === 'number' && expectedCountEl) expectedCountEl.textContent = String(data.expected_count);
            if (typeof data.actual_count === 'number' && actualCountEl) actualCountEl.textContent = String(data.actual_count);
            if (typeof data.absent_count === 'number' && absentCountEl) absentCountEl.textContent = String(data.absent_count);
            if (absentListEl && Array.isArray(data.absent_list)) renderAbsentList(data.absent_list);

            if (typeof data.client_ts === 'number') {
                lastLatencyMs = Date.now() - data.client_ts;
            } else if (typeof lastClientTs === 'number') {
                lastLatencyMs = Date.now() - lastClientTs;
            }
            if (typeof data.processing_ms === 'number') lastProcessingMs = data.processing_ms;

            updatePerfBadges();
            tickFps();
        };

        ws.onerror = () => {
            wsState = 'DISCONNECTED';
            updateWsStatus();
        };

        ws.onclose = () => {
            wsState = 'DISCONNECTED';
            updateWsStatus();
            stopCamera();
        };
    }

    function startScheduler() {
        stopScheduler();
        sendState = 'IDLE';
        const intervalMs = runtimeSettings.capture.frame_interval_ms || 33;
        schedulerTimer = window.setInterval(tickSend, Math.max(15, intervalMs));
    }

    function stopScheduler() {
        if (!schedulerTimer) return;
        window.clearInterval(schedulerTimer);
        schedulerTimer = null;
    }

    function tickSend() {
        if (!isRunning) return;
        if (!ws || ws.readyState !== WebSocket.OPEN || wsState !== 'OPEN') return;

        if (inFlight) {
            if (typeof inFlightSince === 'number' && (Date.now() - inFlightSince) > 5000) {
                inFlight = false;
                inFlightSince = null;
            } else {
                return;
            }
        }
        if (sendState === 'SENDING') {
            if (ws.bufferedAmount === 0) sendState = 'IDLE';
            return;
        }
        if (ws.bufferedAmount !== 0) return;
        if (!video || video.readyState < 2) return;

        const width = runtimeSettings.capture.width || 1280;
        const height = runtimeSettings.capture.height || 720;

        if (!tempCanvas) {
            tempCanvas = document.createElement('canvas');
            tempCanvas.width = width;
            tempCanvas.height = height;
            tempCtx = tempCanvas.getContext('2d');
        }
        if (tempCanvas.width !== width || tempCanvas.height !== height) {
            tempCanvas.width = width;
            tempCanvas.height = height;
        }

        tempCtx.drawImage(video, 0, 0, width, height);
        const jpegQuality = runtimeSettings.capture.jpeg_quality ?? 0.7;
        const dataURL = tempCanvas.toDataURL('image/jpeg', jpegQuality);
        const courseId = courseSelect ? courseSelect.value : '';
        lastClientTs = Date.now();
        ws.send(JSON.stringify({ image: dataURL, course_id: courseId, task_id: activeTaskId, client_ts: lastClientTs }));
        sendState = 'SENDING';
        inFlight = true;
        inFlightSince = lastClientTs;
    }

    async function onCameraChange() {
        if (!isRunning) return;
        try {
            await openStream(cameraSelect ? cameraSelect.value : '');
            await refreshCameraList(true);
        } catch (_) {
            alert('切换摄像头失败，请检查设备权限或设备状态。');
        }
    }

    async function startCamera() {
        try {
            if (!ensureCameraConsent()) return;
            const courseId = courseSelect ? courseSelect.value : '';
            if (!courseId && !activeTaskId) {
                alert('请先选择课程');
                return;
            }
            await loadRuntimeSettings();
            await openStream(cameraSelect ? cameraSelect.value : '');
            await refreshCameraList(true);
            connectWebSocket();
        } catch (err) {
            console.error("Error accessing camera:", err);
            const name = err && err.name ? err.name : '';
            const msg = err && err.message ? err.message : '';
            let tip = "无法访问摄像头。";
            if (name === 'NotAllowedError' || name === 'SecurityError') {
                tip = "摄像头权限被拒绝或浏览器策略限制。请在浏览器地址栏左侧“站点设置”中允许摄像头，然后刷新页面重试。";
            } else if (name === 'NotFoundError') {
                tip = "未检测到可用摄像头设备。请确认摄像头已连接且未被禁用。";
            } else if (name === 'NotReadableError') {
                tip = "摄像头被占用或设备异常。请关闭其它正在使用摄像头的软件（微信/QQ/Teams等）后重试。";
            } else if (window.location.protocol !== 'https:' && window.location.hostname !== 'localhost' && window.location.hostname !== '127.0.0.1') {
                tip = "浏览器要求 HTTPS 或 localhost 才能访问摄像头。请用 http://127.0.0.1:8000 或 http://localhost:8000 打开本系统。";
            }
            alert(`${tip}${name || msg ? `\n\n错误信息：${name}${msg ? ` - ${msg}` : ''}` : ''}`);
        }
    }

    function stopCamera() {
        isRunning = false;
        stopScheduler();
        stopStreamTracks();

        if (ws) {
            try { ws.close(); } catch (_) { }
            ws = null;
        }
        wsState = 'DISCONNECTED';
        sendState = 'IDLE';
        inFlight = false;
        inFlightSince = null;
        lastClientTs = null;
        lastLatencyMs = null;
        lastProcessingMs = null;
        recvFrames = 0;
        lastFpsTick = performance.now();

        if (startBtn) startBtn.disabled = false;
        if (stopBtn) stopBtn.disabled = true;
        ctx.clearRect(0, 0, canvas.width, canvas.height);
        updateWsStatus();
        resetStatsUI();
    }

    function drawResults(faces) {
        ctx.clearRect(0, 0, canvas.width, canvas.height);
        faces.forEach(face => {
            const [x1, y1, x2, y2] = face.box;
            const name = face.name;
            const score = face.score;

            ctx.strokeStyle = name === 'Unknown' ? '#dc3545' : '#198754';
            ctx.lineWidth = 3;
            ctx.strokeRect(x1, y1, x2 - x1, y2 - y1);

            ctx.fillStyle = name === 'Unknown' ? '#dc3545' : '#198754';
            ctx.font = 'bold 18px Arial';
            const label = `${name}  ${(Number.isFinite(score) ? score : 0).toFixed(2)}`;
            const ty = Math.max(18, y1 - 10);
            ctx.fillText(label, x1, ty);
        });
    }

    function updateWsStatus() {
        if (!wsStatusBadge) return;
        if (wsState === 'OPEN') {
            wsStatusBadge.className = 'badge bg-success';
            wsStatusBadge.textContent = 'WS: 已连接';
        } else if (wsState === 'CONNECTING') {
            wsStatusBadge.className = 'badge bg-warning text-dark';
            wsStatusBadge.textContent = 'WS: 连接中';
        } else {
            wsStatusBadge.className = 'badge bg-secondary';
            wsStatusBadge.textContent = 'WS: 断开';
        }
    }

    function tickFps() {
        const now = performance.now();
        if (now - lastFpsTick < 1000) return;
        const fps = Math.round((recvFrames * 1000) / Math.max(1, now - lastFpsTick));
        if (fpsBadge) fpsBadge.textContent = `FPS: ${fps}`;
        recvFrames = 0;
        lastFpsTick = now;
    }

    function updatePerfBadges() {
        if (latencyBadge) latencyBadge.textContent = `延迟: ${lastLatencyMs == null ? '-' : `${Math.max(0, Math.round(lastLatencyMs))}ms`}`;
        if (processingBadge) processingBadge.textContent = `处理: ${lastProcessingMs == null ? '-' : `${Math.max(0, Math.round(lastProcessingMs))}ms`}`;
    }

    function resetStatsUI() {
        if (countBadge) countBadge.textContent = '0';
        if (expectedCountEl) expectedCountEl.textContent = '0';
        if (actualCountEl) actualCountEl.textContent = '0';
        if (absentCountEl) absentCountEl.textContent = '0';
        if (absentListEl) absentListEl.innerHTML = '<tr><td colspan="2" class="text-muted">请选择课程并开始监控</td></tr>';
        if (fpsBadge) fpsBadge.textContent = 'FPS: 0';
        if (latencyBadge) latencyBadge.textContent = '延迟: -';
        if (processingBadge) processingBadge.textContent = '处理: -';
    }

    function renderAbsentList(items) {
        if (!absentListEl) return;
        if (!items.length) {
            absentListEl.innerHTML = '<tr><td colspan="2" class="text-muted">暂无缺勤</td></tr>';
            return;
        }
        absentListEl.innerHTML = '';
        items.forEach(it => {
            const tr = document.createElement('tr');
            const td1 = document.createElement('td');
            const td2 = document.createElement('td');
            td1.textContent = it.student_no || '-';
            td2.textContent = it.name || '-';
            tr.appendChild(td1);
            tr.appendChild(td2);
            absentListEl.appendChild(tr);
        });
    }

    refreshCameraList(true);
    bootstrapTaskFromUrl();
}

const printHistoryBtn = document.getElementById('printHistoryBtn');
if (printHistoryBtn) {
    printHistoryBtn.addEventListener('click', () => window.print());
}

const attendanceDetailModal = document.getElementById('attendanceDetailModal');
if (attendanceDetailModal) {
    attendanceDetailModal.addEventListener('show.bs.modal', (event) => {
        const button = event.relatedTarget;
        if (!button) return;

        const get = (k) => button.getAttribute(k) || '-';

        const sid = document.getElementById('detailStudentId');
        const name = document.getElementById('detailName');
        const cls = document.getElementById('detailClassName');
        const course = document.getElementById('detailCourseName');
        const ts = document.getElementById('detailTimestamp');
        const conf = document.getElementById('detailConfidence');
        const st = document.getElementById('detailStatus');

        if (sid) sid.textContent = get('data-student-id');
        if (name) name.textContent = get('data-name');
        if (cls) cls.textContent = get('data-class-name');
        if (course) course.textContent = get('data-course-name');
        if (ts) ts.textContent = get('data-timestamp');
        if (conf) {
            const raw = button.getAttribute('data-confidence');
            conf.textContent = raw && raw !== 'None' ? raw : '-';
        }
        if (st) st.textContent = get('data-status');
    });
}
