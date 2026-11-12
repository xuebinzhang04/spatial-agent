// "Step through this case" links on task cards open that exact episode in the replay explorer.
(function () {
  document.addEventListener("click", event => {
    const link = event.target.closest && event.target.closest("a.case-link");
    if (!link || !window.SpatialDemo) return;
    const demo = window.SpatialDemo;
    if (demo.chooseTask(link.dataset.task) && demo.chooseAgent("oracle")) demo.chooseEpisode(link.dataset.episode);
  });
})();
