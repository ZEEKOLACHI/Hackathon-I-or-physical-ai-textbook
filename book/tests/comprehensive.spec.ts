import { test, expect } from '@playwright/test';

test.describe('Homepage', () => {
  test('hero section displays correctly', async ({ page }) => {
    await page.goto('/');

    // Check hero title
    const heroTitle = page.locator('.hero__title');
    await expect(heroTitle).toBeVisible();
    await expect(heroTitle).toContainText('Physical AI');

    // Check hero subtitle
    const heroSubtitle = page.locator('.hero__subtitle');
    await expect(heroSubtitle).toBeVisible();

    // Check CTA buttons
    const startButton = page.locator('.hero__buttons .button--primary');
    await expect(startButton).toBeVisible();
    await expect(startButton).toContainText('Start Reading');

    const curriculumButton = page.locator('.hero__buttons .button--secondary');
    await expect(curriculumButton).toBeVisible();
  });

  test('feature cards display correctly', async ({ page }) => {
    await page.goto('/');

    // Check features section exists
    const featuresSection = page.locator('.features');
    await expect(featuresSection).toBeVisible();

    // Check all 3 feature cards are present
    const cards = page.locator('.features .card');
    await expect(cards).toHaveCount(3);

    // Check card content
    await expect(page.locator('.features .card').first()).toContainText('Chapters');
  });

  test('table of contents displays correctly', async ({ page }) => {
    await page.goto('/');

    // Check TOC section
    const tocSection = page.locator('.toc-section');
    await expect(tocSection).toBeVisible();

    // Check "What You'll Learn" heading
    await expect(tocSection.locator('h2')).toContainText("What You'll Learn");

    // Check all 7 parts are listed
    const tocCards = page.locator('.toc-card');
    await expect(tocCards).toHaveCount(7);
  });

  test('navigation to first chapter works', async ({ page }) => {
    await page.goto('/');

    // Click Start Reading button
    await page.click('.hero__buttons .button--primary');

    // Should navigate to first chapter
    await expect(page).toHaveURL(/part-1-foundations\/ch-1-01/);

    // Chapter content should be visible
    await expect(page.locator('article')).toBeVisible();
  });

  test('navbar displays correctly', async ({ page }) => {
    await page.goto('/');

    // Check navbar logo/title
    const navbarTitle = page.locator('.navbar__title');
    await expect(navbarTitle).toBeVisible();

    // Check Chapters link in navbar
    const chaptersLink = page.locator('.navbar__items >> text=Chapters');
    await expect(chaptersLink).toBeVisible();
  });
});

test.describe('Chapters', () => {
  const chapters = [
    { path: '/part-1-foundations/ch-1-01', title: 'Introduction' },
    { path: '/part-1-foundations/ch-1-02', title: 'ROS 2' },
    { path: '/part-1-foundations/ch-1-03', title: 'Simulation' },
  ];

  for (const chapter of chapters) {
    test(`${chapter.title} chapter loads correctly`, async ({ page }) => {
      await page.goto(chapter.path);

      // Page should load
      await expect(page.locator('article')).toBeVisible();

      // Should have heading
      const heading = page.locator('article h1').first();
      await expect(heading).toBeVisible();

      // Sidebar should be visible on desktop
      const sidebar = page.locator('.theme-doc-sidebar-container');
      await expect(sidebar).toBeVisible();
    });
  }

  test('chapter navigation (prev/next) works', async ({ page }) => {
    await page.goto('/part-1-foundations/ch-1-01');

    // Check for pagination nav
    const paginationNav = page.locator('.pagination-nav');
    await expect(paginationNav).toBeVisible();

    // Click next
    const nextLink = page.locator('.pagination-nav__link--next');
    if (await nextLink.isVisible()) {
      await nextLink.click();
      await expect(page).toHaveURL(/ch-1-02/);
    }
  });

  test('sidebar navigation works', async ({ page }) => {
    await page.goto('/part-1-foundations/ch-1-01');

    // Click on a different chapter in sidebar
    const sidebarLink = page.locator('.menu__link:has-text("ROS 2")').first();
    if (await sidebarLink.isVisible()) {
      await sidebarLink.click();
      await expect(page).toHaveURL(/ch-1-02/);
    }
  });
});

test.describe('Responsive Design', () => {
  test('mobile menu toggle works', async ({ page }) => {
    // Set mobile viewport
    await page.setViewportSize({ width: 375, height: 667 });
    await page.goto('/');

    // Check mobile menu toggle is visible
    const menuToggle = page.locator('.navbar__toggle');
    if (await menuToggle.isVisible()) {
      await menuToggle.click();

      // Mobile sidebar should appear
      const mobileSidebar = page.locator('.navbar-sidebar');
      await expect(mobileSidebar).toBeVisible();
    }
  });

  test('homepage is responsive', async ({ page }) => {
    await page.setViewportSize({ width: 375, height: 667 });
    await page.goto('/');

    // Hero should still be visible
    await expect(page.locator('.hero__title')).toBeVisible();

    // Buttons should stack on mobile
    const heroButtons = page.locator('.hero__buttons');
    await expect(heroButtons).toBeVisible();
  });
});

test.describe('No Console Errors', () => {
  test('homepage has no console errors', async ({ page }) => {
    const errors: string[] = [];
    page.on('console', msg => {
      if (msg.type() === 'error') {
        errors.push(msg.text());
      }
    });

    await page.goto('/');
    await page.waitForLoadState('networkidle');

    // Filter out known acceptable errors (like failed API calls in dev)
    const criticalErrors = errors.filter(e =>
      !e.includes('Failed to load resource') &&
      !e.includes('net::ERR')
    );

    expect(criticalErrors).toHaveLength(0);
  });
});
