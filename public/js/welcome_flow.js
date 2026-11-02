(() => {
  function installWelcomeRedirect(documentRef, windowRef) {
    const controllerKey = '__BYAKU_WELCOME_REDIRECT__';
    if (windowRef[controllerKey]) return windowRef[controllerKey];

    const continueLink = documentRef.querySelector('.skip-button');
    if (!continueLink) return () => {};

    let navigated = false;
    let timer = windowRef.setTimeout(() => navigate(), 3000);

    function navigate(event) {
      if (event) event.preventDefault();
      if (navigated) return;
      navigated = true;
      if (timer !== null) windowRef.clearTimeout(timer);
      timer = null;
      windowRef.location.assign(continueLink.href);
    }

    function cancel() {
      if (timer !== null) windowRef.clearTimeout(timer);
      timer = null;
    }

    continueLink.addEventListener('click', navigate);
    windowRef.addEventListener('pagehide', cancel, { once: true });
    windowRef[controllerKey] = cancel;
    return cancel;
  }

  if (typeof window !== 'undefined' && typeof document !== 'undefined') {
    installWelcomeRedirect(document, window);
  }

  if (typeof module !== 'undefined' && module.exports) {
    module.exports = { installWelcomeRedirect };
  }
})();
