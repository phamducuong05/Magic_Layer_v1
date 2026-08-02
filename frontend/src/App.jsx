import { useState } from 'react'

import UploadScreen from './components/UploadScreen'
import Workspace from './components/Workspace'

export default function App() {
  const [canvasData, setCanvasData] = useState(null)

  if (canvasData) {
    return <Workspace data={canvasData} onBack={() => setCanvasData(null)} />
  }

  return <UploadScreen onProcessed={setCanvasData} />
}
