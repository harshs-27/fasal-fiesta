// Fasal Fiesta — Main JS

// Intersection observer for card animations
const observer = new IntersectionObserver((entries) => {
  entries.forEach((entry, i) => {
    if (entry.isIntersecting) {
      setTimeout(() => {
        entry.target.style.opacity = '1';
        entry.target.style.transform = 'translateY(0)';
      }, i * 120);
    }
  });
}, { threshold: 0.1 });

document.querySelectorAll('.module-card').forEach((card, i) => {
  card.style.opacity = '0';
  card.style.transform = 'translateY(30px)';
  card.style.transition = 'opacity 0.6s ease, transform 0.6s cubic-bezier(.2,.8,.2,1), box-shadow 0.35s, border-color 0.3s';
  observer.observe(card);
});

// Particle effect on hero
function createParticle() {
  const p = document.createElement('div');
  p.style.cssText = `
    position:fixed; pointer-events:none; z-index:0;
    width:3px; height:3px; border-radius:50%;
    background: rgba(140,198,63,0.6);
    left:${Math.random()*100}vw;
    top:100vh;
    animation: particle-rise ${4+Math.random()*4}s ease-in forwards;
  `;
  document.body.appendChild(p);
  setTimeout(() => p.remove(), 8000);
}

// Add particle keyframes dynamically
const style = document.createElement('style');
style.textContent = `
  @keyframes particle-rise {
    to { transform: translateY(-110vh) translateX(${Math.random()*100-50}px); opacity: 0; }
  }
`;
document.head.appendChild(style);

// Only run particles on home page
if (document.querySelector('.hero')) {
  setInterval(createParticle, 800);
}

// Strip ticker effect
const strip = document.querySelector('.feature-strip');
if (strip) {
  strip.addEventListener('mouseenter', () => {
    strip.style.background = 'rgba(212,135,10,0.08)';
  });
  strip.addEventListener('mouseleave', () => {
    strip.style.background = 'rgba(212,135,10,0.04)';
  });
}
