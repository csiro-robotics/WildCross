$(document).ready(function() {
    $(".navbar-burger").click(function() {
      $(".navbar-burger").toggleClass("is-active");
      $(".navbar-menu").toggleClass("is-active");
    });

    if (document.querySelector('.carousel')) {
      bulmaCarousel.attach('.carousel', {
        slidesToScroll: 1,
        slidesToShow: 1,
        loop: true,
        infinite: true
      });
    }

    var chartEl = document.getElementById('vpr-results-chart');
    if (chartEl && window.Plotly) {
      var years = [2016, 2023, 2024, 2024];
      var recall = [55.63, 57.77, 61.41, 61.87];
      var labels = ['NetVLAD', 'MixVPR', 'SALAD', 'BoQ'];

      var trace = {
        x: years,
        y: recall,
        mode: 'lines+markers+text',
        type: 'scatter',
        text: labels,
        textposition: ['top center', 'top center', 'bottom left', 'top right'],
        line: {color: '#00a6a6', width: 3},
        marker: {color: '#00a6a6', size: 10},
        hovertemplate: '%{text}<br>Year: %{x}<br>Recall@1: %{y:.2f}%<extra></extra>'
      };

      var layout = {
        title: {text: 'Inter-Sequence VPR Results on WildCross', x: 0.5},
        xaxis: {
          title: 'Publication Year',
          tickmode: 'array',
          tickvals: [2016, 2023, 2024]
        },
        yaxis: {
          title: 'Recall@1 (%)',
          range: [50, 65]
        },
        margin: {l: 70, r: 20, t: 60, b: 70},
        paper_bgcolor: 'white',
        plot_bgcolor: 'white',
        showlegend: false
      };

      Plotly.newPlot(chartEl, [trace], layout, {responsive: true, displayModeBar: false});
    }
});
