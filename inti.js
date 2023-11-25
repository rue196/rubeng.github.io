  function toggleDropdown() {
  var dropdown = document.getElementById("myDropdown");
  dropdown.style.display = (dropdown.style.display === 'block') ? 'none' : 'block';
}

$(index.html).ready(function () {
  if (!$.browser.webkit) {
      $('.wrapper').html('<p>Sorry! Non webkit users. :(</p>');
  }
});
