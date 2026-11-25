// Click a cover or animated preview to replace it with the full MP4 in place.
// Without JavaScript the preview is a plain link to the MP4.
document.querySelectorAll(".player").forEach((player) => {
  const link = player.querySelector("a");
  link.addEventListener("click", (event) => {
    event.preventDefault();
    const video = document.createElement("video");
    video.src = player.dataset.src;
    video.controls = true;
    video.autoplay = true;
    video.playsInline = true;
    video.preload = "auto";
    const img = player.querySelector("img");
    video.setAttribute("aria-label", link.getAttribute("aria-label") || (img ? img.alt : "Environment video"));
    player.replaceChildren(video);
    video.focus();
  });
});

// Copy buttons for code blocks.
document.querySelectorAll(".copy").forEach((button) => {
  button.addEventListener("click", async () => {
    const text = document.getElementById(button.dataset.copy).innerText;
    try {
      await navigator.clipboard.writeText(text);
      button.textContent = "Copied";
    } catch {
      button.textContent = "Select & copy";
    }
    setTimeout(() => { button.textContent = "Copy"; }, 1600);
  });
});
