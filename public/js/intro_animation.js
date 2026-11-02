(() => {
  const intro = document.querySelector('#intro-screen');
  const skip = document.querySelector('#intro-skip');
  const fireVideo = document.querySelector('#intro-fire-video');
  if (!intro || !skip) return;

  const motionPreference = window.matchMedia('(prefers-reduced-motion: reduce)');
  const reducedMotion = motionPreference.matches;
  const userInterface = [...document.querySelectorAll('.user-header, .user-main')];
  const introDuration = reducedMotion ? 0 : 3000;
  const transitionDuration = reducedMotion ? 100 : 560;
  let remaining = introDuration;
  let startedAt = 0;
  let dismissTimer = 0;

  function dismiss() {
    if (intro.classList.contains('is-leaving')) return;
    fireVideo?.pause();
    intro.classList.add('is-leaving');
    intro.setAttribute('aria-hidden', 'true');
    skip?.setAttribute('tabindex', '-1');
    window.clearTimeout(dismissTimer);
    window.setTimeout(() => {
      intro.remove();
      document.body.classList.add('intro-finished');
      userInterface.forEach((element) => { element.inert = false; });
      document.querySelector('#start-listening')?.focus({ preventScroll: true });
    }, transitionDuration);
  }

  function resumeIntro() {
    if (!intro.isConnected || intro.classList.contains('is-leaving')) return;
    if (!motionPreference.matches && document.visibilityState === 'visible') {
      void fireVideo?.play().catch(() => {});
    }
    startedAt = performance.now();
    dismissTimer = window.setTimeout(dismiss, remaining);
  }

  function pauseIntro() {
    if (!startedAt) return;
    remaining = Math.max(0, remaining - (performance.now() - startedAt));
    startedAt = 0;
    window.clearTimeout(dismissTimer);
    fireVideo?.pause();
  }

  userInterface.forEach((element) => { element.inert = true; });
  window.setTimeout(() => intro.classList.add('title-revealed'), reducedMotion ? 0 : 1750);
  skip?.addEventListener('click', dismiss);
  document.addEventListener('visibilitychange', () => {
    if (document.visibilityState === 'hidden') {
      pauseIntro();
      return;
    }
    resumeIntro();
  });
  motionPreference.addEventListener?.('change', (event) => {
    if (event.matches) fireVideo?.pause();
    else if (document.visibilityState === 'visible') void fireVideo?.play().catch(() => {});
  });
  document.addEventListener('keydown', (event) => {
    if (event.key === 'Escape' && intro.isConnected) dismiss();
  });
  if (document.visibilityState === 'visible') resumeIntro();
})();
