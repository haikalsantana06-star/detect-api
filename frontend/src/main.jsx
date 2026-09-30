import { StrictMode } from 'react'
import { createRoot } from 'react-dom/client'
import './index.css'
import AppStats from './AppStats.jsx'

createRoot(document.getElementById('root')).render(
  <StrictMode>
    <AppStats />
  </StrictMode>,
)
