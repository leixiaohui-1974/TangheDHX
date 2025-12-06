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
        { label: 'Gate 1 Flow', borderColor: 'blue', data: [] },
        { label: 'Gate 2 Flow', borderColor: 'red', data: [] },
        { label: 'Gate 3 Flow', borderColor: 'green', data: [] }
    ]},
    options: { ...commonOptions, plugins: { title: { display: true, text: 'Flow Rate (m³/s)' } } }
});

const openingChart = new Chart(ctxOpen, {
    type: 'line',
    data: { labels: [], datasets: [
        { label: 'Gate 1 Opening', borderColor: 'blue', data: [] },
        { label: 'Gate 2 Opening', borderColor: 'red', data: [] },
        { label: 'Gate 3 Opening', borderColor: 'green', data: [] }
    ]},
    options: { ...commonOptions, plugins: { title: { display: true, text: 'Gate Opening (m)' } } }
});

const vibChart = new Chart(ctxVib, {
    type: 'line',
    data: { labels: [], datasets: [
        { label: 'Gate 1 Vib (g)', borderColor: 'blue', data: [] },
        { label: 'Gate 2 Vib (g)', borderColor: 'red', data: [] },
        { label: 'Gate 3 Vib (g)', borderColor: 'green', data: [] }
    ]},
    options: { ...commonOptions, plugins: { title: { display: true, text: 'Vibration Acceleration (g)' } } }
});

const freqChart = new Chart(ctxFreq, {
    type: 'line',
    data: { labels: [], datasets: [
        { label: 'Gate 1 Freq', borderColor: 'blue', data: [] },
        { label: 'Gate 2 Freq', borderColor: 'red', data: [] },
        { label: 'Gate 3 Freq', borderColor: 'green', data: [] },
        { label: 'Structure Freq (2.8Hz)', borderColor: 'black', borderDash: [5, 5], data: [] }
    ]},
    options: { ...commonOptions, plugins: { title: { display: true, text: 'Vortex Frequency (Hz)' } } }
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

function fetchData() {
    fetch('/api/state')
        .then(response => response.json())
        .then(data => {
            document.getElementById('totalFlowDisplay').innerText = data.total_flow.toFixed(2);
            document.getElementById('statusText').innerText = 'Scenario: ' + (data.active_scenario || 'Normal');

            updateChart(flowChart, data.flows);
            updateChart(openingChart, data.openings);
            updateChart(vibChart, data.vibrations);

            // Add Structure Freq line to dataset 3
            const freqs = [...data.frequencies, 2.8];
            updateChart(freqChart, freqs);
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

setInterval(fetchData, 200);
