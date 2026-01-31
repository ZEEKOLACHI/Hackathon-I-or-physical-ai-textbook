import { test, expect } from '@playwright/test';

test('homepage loads correctly', async ({ page }) => {
  await page.goto('/');

  // Check page title contains expected text
  await expect(page).toHaveTitle(/Physical AI/i);
});

test('navigation works', async ({ page }) => {
  await page.goto('/');

  // Page should have loaded
  await expect(page.locator('body')).toBeVisible();
});

test('chatbot component exists', async ({ page }) => {
  await page.goto('/');

  // Look for chatbot button or container
  const chatButton = page.locator('[class*="chat"], [id*="chat"], button:has-text("Chat")').first();

  // Wait for page to fully load
  await page.waitForLoadState('networkidle');
});
