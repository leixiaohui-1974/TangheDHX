const ctxFlow = document.getElementById('flowChart').getContext('2d');
const ctxOpen = document.getElementById('openingChart').getContext('2d');
const ctxVib = document.getElementById('vibChart').getContext('2d');
const ctxFreq = document.getElementById('freqChart').getContext('2d');

// Chart Setup
const commonOptions = {
    responsive: true,
    maintainAspectRatio: false,
    animation: false,
    elements: { point: { radius: 0 } },
    scales: { x: { display: false } }
};

const flowChart = new Chart(ctxFlow, {
    type: 'line',
    data: { labels: [], datasets: [
        { label: 'G1 Flow', borderColor: 'blue', data: [] },
        { label: 'G2 Flow', borderColor: 'red', data: [] },
        { label: 'G3 Flow', borderColor: 'green', data: [] }
    ]},
    options: { ...commonOptions, plugins: { title: { display: true, text: 'Flow Rate (m³/s)' } } }
});

const openingChart = new Chart(ctxOpen, {
    type: 'line',
    data: { labels: [], datasets: [
        { label: 'G1 Open', borderColor: 'blue', data: [] },
        { label: 'G2 Open', borderColor: 'red', data: [] },
        { label: 'G3 Open', borderColor: 'green', data: [] }
    ]},
    options: { ...commonOptions, plugins: { title: { display: true, text: 'Gate Opening (m)' } } }
});

const vibChart = new Chart(ctxVib, {
    type: 'line',
    data: { labels: [], datasets: [
        { label: 'G1 Vib', borderColor: 'blue', data: [] },
        { label: 'G2 Vib', borderColor: 'red', data: [] },
        { label: 'G3 Vib', borderColor: 'green', data: [] }
    ]},
    options: { ...commonOptions, plugins: { title: { display: true, text: 'Vibration (g)' } } }
});

const freqChart = new Chart(ctxFreq, {
    type: 'line',
    data: { labels: [], datasets: [
        { label: 'G1 Freq', borderColor: 'blue', data: [] },
        { label: 'G2 Freq', borderColor: 'red', data: [] },
        { label: 'G3 Freq', borderColor: 'green', data: [] },
        { label: 'Limit (2.8Hz)', borderColor: 'black', borderDash: [5, 5], data: [] }
    ]},
    options: { ...commonOptions, plugins: { title: { display: true, text: 'Vortex Freq (Hz)' } } }
});

const MAX_POINTS = 50;

function updateChart(chart, newData) {
    if (chart.data.labels.length > MAX_POINTS) {
        chart.data.labels.shift();
        chart.data.datasets.forEach(ds => ds.data.shift());
    }
    chart.data.labels.push('');
    newData.forEach((val, i) => {
        if (chart.data.datasets[i]) {
            chart.data.datasets[i].data.push(val);
        }
    });
    chart.update();
}

function renderGates(openings) {
    const container = document.getElementById('gateSchematic');
    container.innerHTML = ''; // Clear

    openings.forEach((open, i) => {
        const h = open * 30; // Scale factor

        // Simpler: Just a bar representing the gate position.
        // 0 opening = Bar at bottom. 5m opening = Bar at top.

        const bar = document.createElement('div');
        bar.style.width = '60px';
        bar.style.background = '#7f8c8d';
        bar.style.height = '150px';
        bar.style.position = 'absolute';
        bar.style.bottom = (open * 30) + 'px'; // Moves up
        bar.style.left = (50 + i * 100) + 'px';
        bar.style.border = '2px solid #34495e';

        const water = document.createElement('div');
        water.style.width = '60px';
        water.style.height = (open * 30) + 'px';
        water.style.background = 'rgba(52, 152, 219, 0.7)';
        water.style.position = 'absolute';
        water.style.bottom = '0';
        water.style.left = (50 + i * 100) + 'px';

        const label = document.createElement('div');
        label.innerText = `G${i+1}: ${open.toFixed(2)}m`;
        label.style.position = 'absolute';
        label.style.bottom = '-20px';
        label.style.left = (50 + i * 100) + 'px';
        label.style.width = '60px';
        label.style.textAlign = 'center';

        container.appendChild(bar);
        container.appendChild(water);
        container.appendChild(label);
    });
}

const loggedEvents = new Set();

function addLog(msg) {
    if (loggedEvents.has(msg)) return; // Simple dedup
    const logPanel = document.getElementById('eventLogs');
    const entry = document.createElement('div');
    entry.className = 'log-entry';
    if (msg.includes("ALARM")) entry.classList.add('log-ALARM');
    entry.innerText = new Date().toLocaleTimeString() + " - " + msg;
    logPanel.prepend(entry);
    loggedEvents.add(msg);
    if (loggedEvents.size > 20) loggedEvents.clear(); // Reset dedup occasionally
}

function fetchData() {
    fetch('/api/state')
        .then(response => response.json())
        .then(data => {
            document.getElementById('totalFlowDisplay').innerText = data.total_flow.toFixed(2);
            document.getElementById('headLossDisplay').innerText = data.trash_rack_loss.toFixed(2) + " m";
            document.getElementById('detectedScen').innerText = data.supervisor.detected_scenario;

            // Mode Badge
            const badge = document.getElementById('sysMode');
            const mode = data.supervisor.mode;
            badge.innerText = mode;
            badge.className = 'mode-badge mode-' + mode;

            // Charts
            updateChart(flowChart, data.flows);
            updateChart(openingChart, data.openings);
            updateChart(vibChart, data.vibrations);
            const freqs = [...data.frequencies, 2.8];
            updateChart(freqChart, freqs);

            // Schematic
            renderGates(data.openings);

            // Logs
            if (data.supervisor.latest_event) {
                addLog(data.supervisor.latest_event);
            }
            data.supervisor.alarms.forEach(a => addLog("ALARM: " + a));
        });
}

function setFlow() {
    const val = document.getElementById('targetFlowInput').value;
    fetch('/api/control', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ target_flow: val })
    });
}

function setScenario(name) {
    fetch('/api/control', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ scenario: name })
    });
}

function injectFault(type) {
    fetch('/api/control', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ inject_fault: type })
    });
}

setInterval(fetchData, 200);
