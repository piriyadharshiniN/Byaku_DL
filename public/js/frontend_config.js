window.BYAKU_CONFIG = Object.freeze({
  appName: 'BYAKU',
  tagline: 'SEE BEYOND. UNDERSTAND EVERYTHING.',
});

document.querySelectorAll('[data-app-name]').forEach((element) => {
  element.textContent = window.BYAKU_CONFIG.appName;
});
document.querySelectorAll('[data-app-tagline]').forEach((element) => {
  element.textContent = window.BYAKU_CONFIG.tagline;
});
