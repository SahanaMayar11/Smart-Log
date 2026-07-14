(function () {
  const filterBar = document.getElementById("severity-filters");
  if (!filterBar) return;

  const rows = Array.from(document.querySelectorAll("#alerts-table tbody tr"));

  filterBar.addEventListener("click", (event) => {
    const btn = event.target.closest(".chip");
    if (!btn) return;

    filterBar.querySelectorAll(".chip").forEach((c) => c.classList.remove("active"));
    btn.classList.add("active");

    const severity = btn.dataset.severity;
    rows.forEach((row) => {
      const show = severity === "ALL" || row.dataset.severity === severity;
      row.style.display = show ? "" : "none";
    });
  });
})();
