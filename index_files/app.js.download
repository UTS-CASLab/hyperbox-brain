// Content Page Shortcuts
const shortcutsTarget = $('#shortcuts');
if (shortcutsTarget.length > 0) {
  $('.content-container h2, .content-container h3').map(function(idx, el) {
    const title = el.textContent;
    // transforms title into snake-case
    const elTitle = title.replace(/\s/g, '-').toLowerCase();
    // Gets the element type (e.g. h2, h3)
    const elType = $(el).get(0).tagName;
    // Adds snake-case title as an id attribute to target element
    $(el).attr('id', elTitle);
    shortcutsTarget.append(`<div id="${elTitle}-shortcut" class="shortcuts-${elType}">${title}</div>`);

    $(`#${elTitle}-shortcut`).click(function() {
      $([document.documentElement, document.body]).animate({
        scrollTop: $(`#${elTitle}`).offset().top-60
    }, 1000);
    })
  });
}

// Removes the shortcuts container if no shortcuts exist.
// Also removes the 'Get Help' link.
if ($('#shortcuts div').length < 1) {
  $('.shortcuts-container').css('display', 'none');
}