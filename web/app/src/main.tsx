import React, { Suspense, lazy } from 'react'
import ReactDOM from 'react-dom/client'
import { BrowserRouter, Routes, Route } from 'react-router-dom'
import { Layout } from './components/Layout'
import { Home } from './components/pages/Home'
import { About } from './components/pages/About'
import { Privacy } from './components/pages/Privacy'
import './index.css'

// Dev/benchmark-only tooling: lazy-loaded into its own chunk and only
// registered in development so it never ships in the production bundle.
// Gating the `lazy()` call itself behind `import.meta.env.DEV` lets Rollup
// drop both the route and the dynamic-import chunk from prod builds.
const benchmarkRoute = import.meta.env.DEV
    ? (() => {
          const Benchmark = lazy(() =>
              import('./components/pages/Benchmark').then(m => ({ default: m.Benchmark }))
          )
          return (
              <Route
                  path="/benchmark"
                  element={
                      <Suspense fallback={null}>
                          <Benchmark />
                      </Suspense>
                  }
              />
          )
      })()
    : null

ReactDOM.createRoot(document.getElementById('root')!).render(
    <React.StrictMode>
        <BrowserRouter>
            <Routes>
                <Route element={<Layout />}>
                    <Route path="/" element={<Home />} />
                    <Route path="/about" element={<About />} />
                    <Route path="/privacy" element={<Privacy />} />
                    {benchmarkRoute}
                </Route>
            </Routes>
        </BrowserRouter>
    </React.StrictMode>,
)

