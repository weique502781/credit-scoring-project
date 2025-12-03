$(document).ready(function() {
    console.log('Credit Scoring System initialized');

    // Initialize tooltips
    var tooltipTriggerList = [].slice.call(document.querySelectorAll('[data-bs-toggle="tooltip"]'));
    var tooltipList = tooltipTriggerList.map(function (tooltipTriggerEl) {
        return new bootstrap.Tooltip(tooltipTriggerEl);
    });

    // Initialize popovers
    var popoverTriggerList = [].slice.call(document.querySelectorAll('[data-bs-toggle="popover"]'));
    var popoverList = popoverTriggerList.map(function (popoverTriggerEl) {
        return new bootstrap.Popover(popoverTriggerEl);
    });

    // Auto-dismiss alerts after 5 seconds
    $('.alert').delay(5000).fadeOut('slow');

    // Form validation enhancement
    $('form').on('submit', function() {
        // Show loading state
        const submitBtn = $(this).find('button[type="submit"]');
        if (submitBtn.length) {
            submitBtn.prop('disabled', true);
            submitBtn.html('<span class="spinner-border spinner-border-sm" role="status" aria-hidden="true"></span> Processing...');
        }
    });

    // Feature value range indicators
    $('.range-indicator').each(function() {
        const $this = $(this);
        const currentVal = parseFloat($this.data('current'));
        const minVal = parseFloat($this.data('min'));
        const maxVal = parseFloat($this.data('max'));

        if (!isNaN(currentVal) && !isNaN(minVal) && !isNaN(maxVal)) {
            const percentage = ((currentVal - minVal) / (maxVal - minVal)) * 100;
            $this.find('.range-fill').css('width', percentage + '%');
        }
    });

    // Model comparison chart
    if ($('#modelComparisonChart').length) {
        initializeModelComparisonChart();
    }

    // Probability visualization
    $('.probability-visualization').each(function() {
        const $this = $(this);
        const goodProb = parseFloat($this.data('good-prob'));
        const badProb = parseFloat($this.data('bad-prob'));

        if (!isNaN(goodProb) && !isNaN(badProb)) {
            // Animate probability bars
            setTimeout(() => {
                $this.find('.good-prob-fill').css('width', (goodProb * 100) + '%');
                $this.find('.bad-prob-fill').css('width', (badProb * 100) + '%');
            }, 500);
        }
    });
});

function initializeModelComparisonChart() {
    // This function would be populated with actual chart data
    // For now, it's a placeholder
    console.log('Initializing model comparison chart');
}

// API helper functions
async function makePredictionAPI(features, model = 'custom_adaboost') {
    try {
        const response = await fetch('/api/predict', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({
                features: features,
                model: model
            })
        });

        if (!response.ok) {
            throw new Error(`HTTP error! status: ${response.status}`);
        }

        return await response.json();
    } catch (error) {
        console.error('Error making prediction:', error);
        throw error;
    }
}

async function getModelList() {
    try {
        const response = await fetch('/api/models');
        if (!response.ok) {
            throw new Error(`HTTP error! status: ${response.status}`);
        }
        return await response.json();
    } catch (error) {
        console.error('Error getting model list:', error);
        throw error;
    }
}

// Utility functions
function formatPercentage(value) {
    return (value * 100).toFixed(2) + '%';
}

function getRiskColor(probability) {
    if (probability >= 0.7) return 'danger';
    if (probability >= 0.4) return 'warning';
    return 'success';
}

function updateProbabilityVisualization(goodProb, badProb) {
    // Update the probability bars with animation
    $('.good-prob-fill').css({
        'width': (goodProb * 100) + '%',
        'background-color': getRiskColor(badProb) === 'success' ? '#27ae60' : '#e74c3c'
    });

    $('.bad-prob-fill').css({
        'width': (badProb * 100) + '%',
        'background-color': getRiskColor(badProb) === 'danger' ? '#e74c3c' : '#27ae60'
    });

    // Update text
    $('.good-prob-text').text(formatPercentage(goodProb));
    $('.bad-prob-text').text(formatPercentage(badProb));

    // Update risk indicator
    const riskLevel = getRiskColor(badProb);
    const riskText = riskLevel === 'danger' ? 'High Risk' :
                    riskLevel === 'warning' ? 'Medium Risk' : 'Low Risk';

    $('.risk-indicator')
        .removeClass('bg-success bg-warning bg-danger')
        .addClass('bg-' + riskLevel)
        .text(riskText);
}