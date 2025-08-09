// Simple index.js file for website functionality
$(document).ready(function() {
    console.log('GenMech website loaded');

    // Initialize any sliders if present
    if (typeof bulmaSlider !== 'undefined') {
        try {
            bulmaSlider.attach();
        } catch (e) {
            console.log('Slider initialization skipped');
        }
    }

    // Initialize any carousels if present
    if (typeof bulmaCarousel !== 'undefined') {
        try {
            bulmaCarousel.attach();
        } catch (e) {
            console.log('Carousel initialization skipped');
        }
    }

    // Basic smooth scrolling for anchor links
    $('a[href^="#"]').on('click', function(event) {
        var target = $(this.getAttribute('href'));
        if (target.length) {
            event.preventDefault();
            $('html, body').stop().animate({
                scrollTop: target.offset().top - 50
            }, 1000);
        }
    });
});
