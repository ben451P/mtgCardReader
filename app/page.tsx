"use client"

import { useState, useCallback } from "react"
import { UploadPage } from "@/components/upload-page"
import { LoadingPage } from "@/components/loading-page"
import { ResultsPage } from "@/components/results-page"

type AppState = "upload" | "loading" | "results"

interface PriceStats {
  collector_num: string
  price: string
  price_foil: string
  rarity: string
}

export default function Home() {
  const [appState, setAppState] = useState<AppState>("upload")
  const [selectedFile, setSelectedFile] = useState<File | null>(null)
  const [previewUrl, setPreviewUrl] = useState<string | null>(null)
  const [stats, setStats] = useState<PriceStats | null>(null)

  const handleFileSelect = useCallback((file: File) => {
    setSelectedFile(file)
    const url = URL.createObjectURL(file)
    setPreviewUrl(url)
  }, [])

  const handleFindPrice = useCallback(async () => {
    if (!selectedFile) return

    setAppState("loading")

    const formData = new FormData()
    formData.append("image", selectedFile)

    try {
      const response = await fetch("http://localhost:5000/api/fetch_price", {
        method: "POST",
        body: formData,
      })

      if (!response.ok) {
        throw new Error("Unknown failure while analyzing image.")
      }

      const data = await response.json()
      
      if (data.response == "card_read_error") throw new Error("Could not read the card: Please get a clearer picture of the card")

      setStats(data)
      setAppState("results")
    } catch (error) {
      console.error("[v0] Error analyzing image:", error)
      setAppState("upload")
    }
  }, [selectedFile])

  const handleReset = useCallback(() => {
    setSelectedFile(null)
    setPreviewUrl(null)
    setStats(null)
    setAppState("upload")
  }, [])

  return (
    <main className="min-h-screen bg-background">
      {appState === "upload" && (
        <UploadPage
          onFileSelect={handleFileSelect}
          onFindPrice={handleFindPrice}
          selectedFile={selectedFile}
          previewUrl={previewUrl}
        />
      )}
      {appState === "loading" && <LoadingPage />}
      {appState === "results" && stats && (
        <ResultsPage stats={stats} previewUrl={previewUrl} onReset={handleReset} />
      )}
    </main>
  )
}
