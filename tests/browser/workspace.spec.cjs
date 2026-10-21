const { test, expect } = require('@playwright/test');
const AxeBuilder = require('@axe-core/playwright').default;

async function ready(page) {
  await page.goto('/');
  await expect(page.getByRole('heading', { name: 'Follow the evidence.' })).toBeVisible();
  await expect(page.locator('[data-testid="stApp"]')).toHaveAttribute('data-test-script-state', 'notRunning');
}
async function accessible(page, testInfo, name) {
  await expect(page.locator('[data-testid="stApp"]')).toHaveAttribute('data-test-script-state', 'notRunning');
  const results = await new AxeBuilder({ page }).withTags(['wcag2a', 'wcag2aa', 'wcag21a', 'wcag21aa']).analyze();
  await testInfo.attach(name, { body: JSON.stringify(results, null, 2), contentType: 'application/json' });
  expect(results.violations).toEqual([]);
}

test('desktop investigation, evidence, report download, and WCAG checks', async ({ page }, testInfo) => {
  await page.setViewportSize({ width: 1280, height: 900 });
  await ready(page);
  await accessible(page, testInfo, 'initial-accessibility');
  await page.screenshot({ path: testInfo.outputPath('desktop-initial.png'), fullPage: true });
  const question = page.getByRole('textbox', { name: 'What would you like to investigate?' });
  await question.fill('Investigate customer churn');
  await page.getByRole('button', { name: 'Start investigation', exact: true }).click();
  await expect(page.getByText('Investigation complete · 5 analyses', { exact: true })).toBeVisible();
  await expect(page.getByRole('heading', { name: /A01/ })).toBeVisible();
  await expect(page.locator('main img').first()).toBeVisible();
  await accessible(page, testInfo, 'findings-accessibility');
  await page.screenshot({ path: testInfo.outputPath('desktop-findings.png'), fullPage: true });
  await page.getByRole('tab', { name: 'Report & export' }).click();
  const download = page.waitForEvent('download');
  await page.getByRole('button', { name: 'Download investigation bundle' }).click();
  await (await download).saveAs(testInfo.outputPath('investigation.zip'));
});

for (const width of [320, 390, 768]) {
  test(`responsive ${width}px layout, form navigation, and WCAG checks`, async ({ page }, testInfo) => {
    await page.setViewportSize({ width, height: 844 });
    await ready(page);
    expect(await page.evaluate(() => document.documentElement.scrollWidth <= window.innerWidth)).toBeTruthy();
    await page.screenshot({ path: testInfo.outputPath(`workspace-${width}-top.png`), fullPage: true });
    const question = page.getByRole('textbox', { name: 'What would you like to investigate?' });
    await question.focus();
    await page.keyboard.press('Tab');
    expect(await page.evaluate(() => !['BODY', 'HTML'].includes(document.activeElement.tagName))).toBeTruthy();
    await accessible(page, testInfo, `accessibility-${width}`);
    await page.screenshot({ path: testInfo.outputPath(`workspace-${width}.png`), fullPage: true });
  });
}

test('mobile findings, chart descriptions, and statistical review', async ({ page }, testInfo) => {
  await page.setViewportSize({ width: 390, height: 844 });
  await ready(page);
  await page.getByRole('button', { name: 'Start investigation', exact: true }).click();
  await expect(page.getByText('Investigation complete · 5 analyses', { exact: true })).toBeVisible();
  const image = page.locator('main img').first();
  await expect(image).toHaveAttribute('alt', /Numerical values are in the evidence table/);
  await image.scrollIntoViewIfNeeded();
  await accessible(page, testInfo, 'mobile-findings-accessibility');
  await page.screenshot({ path: testInfo.outputPath('mobile-findings.png'), fullPage: true });
  expect(await page.evaluate(() => document.documentElement.scrollWidth <= window.innerWidth)).toBeTruthy();
});
