// Show a spinner on the submit button while the server computes the result:
// when the form is submitted, or when an example link is clicked.

const form = document.querySelector("form[data-loading-form]");

if (form) {
  const button = form.querySelector("button[type=submit]");
  const originalHtml = button.innerHTML;

  function showLoading() {
    button.disabled = true;
    button.innerHTML =
      '<span class="spinner-border spinner-border-sm me-2" aria-hidden="true"></span>' +
      '<span role="status">Disambiguating…</span>';
  }

  form.addEventListener("submit", showLoading);
  document.querySelectorAll("a[data-loading-link]").forEach((link) => {
    link.addEventListener("click", showLoading);
  });

  // Going back restores the page from the browser cache: reset the button.
  window.addEventListener("pageshow", () => {
    button.disabled = false;
    button.innerHTML = originalHtml;
  });
}
