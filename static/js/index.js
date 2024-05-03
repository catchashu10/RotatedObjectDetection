window.HELP_IMPROVE_VIDEOJS = false;


$(document).ready(function() {
    // Check for click events on the navbar burger icon

    var options = {
			slidesToScroll: 1,
			slidesToShow: 1,
			loop: true,
			infinite: true,
			autoplay: true,
			autoplaySpeed: 5000,
    }

		// Initialize all div with carousel class
    var carousels = bulmaCarousel.attach('.carousel', options);
	
    bulmaSlider.attach();

})

document.addEventListener('DOMContentLoaded', function() {
    const gridContainer = document.getElementById('results-grid');
    const imageFolder = 'static/images/outputs/'; // Path to the folder

    for (let i = 1; i <= 36; i++) {
      const img = document.createElement('img');
      img.src = `${imageFolder}${i}.png`;
      img.alt = `Image ${i}`;
      img.className = 'grid-image';
      gridContainer.appendChild(img);
    }
  });