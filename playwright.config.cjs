const { defineConfig } = require('@playwright/test');
const port = process.env.PICO_BROWSER_PORT || '8501';
module.exports = defineConfig({
  testDir: './tests/browser',
  timeout: 60000,
  expect: { timeout: 15000 },
  workers: 1,
  reporter: [['list'], ['html', { open: 'never' }]],
  use: { baseURL: `http://127.0.0.1:${port}`, trace: 'retain-on-failure', screenshot: 'only-on-failure' },
  webServer: {
    command: `${process.env.PICO_PYTHON || 'python3'} -m streamlit run app.py --server.headless=true --server.address=127.0.0.1 --server.port=${port}`,
    url: `http://127.0.0.1:${port}/_stcore/health`,
    reuseExistingServer: !process.env.CI,
    timeout: 60000,
  },
});
