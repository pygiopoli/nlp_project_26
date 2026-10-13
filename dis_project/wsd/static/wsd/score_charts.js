// Draw a horizontal bar chart of the candidate scores for every
// <canvas data-chart-source="ID">, where ID is the id of a
// <script type="application/json"> block produced by Django's json_script.
// The first candidate is the chosen sense and gets the primary color.

const BAR_HEIGHT_PX = 26;      // vertical space per candidate
const CHART_PADDING_PX = 40;   // room for the x axis

function cssVariable(name) {
  return getComputedStyle(document.documentElement).getPropertyValue(name).trim();
}

function drawScoreChart(canvas) {
  const source = document.getElementById(canvas.dataset.chartSource);
  const data = JSON.parse(source.textContent);

  // Chart.js fills the parent, so the parent's height sets the chart height.
  canvas.parentElement.style.height =
    `${data.labels.length * BAR_HEIGHT_PX + CHART_PADDING_PX}px`;

  const chosenColor = cssVariable("--bs-primary");
  const otherColor = cssVariable("--bs-gray-400");
  // Integer ticks for counts (Lesk); automatic ticks for decimal scores.
  const allIntegers = data.scores.every(Number.isInteger);

  new Chart(canvas, {
    type: "bar",
    data: {
      labels: data.labels,
      datasets: [{
        label: "Score",
        data: data.scores,
        backgroundColor: data.scores.map((_, i) => (i === 0 ? chosenColor : otherColor)),
      }],
    },
    options: {
      indexAxis: "y",              // horizontal bars: sense ids on the vertical axis
      maintainAspectRatio: false,  // use the height set above
      plugins: { legend: { display: false } },
      scales: { x: { beginAtZero: true, ticks: { precision: allIntegers ? 0 : undefined } } },
    },
  });
}

document.querySelectorAll("canvas[data-chart-source]").forEach(drawScoreChart);
