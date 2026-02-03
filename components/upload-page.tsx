"use client"

import React from "react"

import { useCallback, useRef, useState } from "react"
import { Upload, ImageIcon } from "lucide-react"
import { Button } from "@/components/ui/button"
import { Card } from "@/components/ui/card"

interface UploadPageProps {
  onFileSelect: (file: File) => void
  onFindPrice: () => void
  selectedFile: File | null
  previewUrl: string | null
}

export function UploadPage({
  onFileSelect,
  onFindPrice,
  selectedFile,
  previewUrl,
}: UploadPageProps) {
  const [isDragging, setIsDragging] = useState(false)
  const fileInputRef = useRef<HTMLInputElement>(null)

  const handleDragOver = useCallback((e: React.DragEvent) => {
    e.preventDefault()
    setIsDragging(true)
  }, [])

  const handleDragLeave = useCallback((e: React.DragEvent) => {
    e.preventDefault()
    setIsDragging(false)
  }, [])

  const handleDrop = useCallback(
    (e: React.DragEvent) => {
      e.preventDefault()
      setIsDragging(false)
      const file = e.dataTransfer.files[0]
      if (file && file.type.startsWith("image/")) {
        onFileSelect(file)
      }
    },
    [onFileSelect]
  )

  const handleFileChange = useCallback(
    (e: React.ChangeEvent<HTMLInputElement>) => {
      const file = e.target.files?.[0]
      if (file) {
        onFileSelect(file)
      }
    },
    [onFileSelect]
  )

  const handleUploadClick = useCallback(() => {
    fileInputRef.current?.click()
  }, [])

  return (
    <div className="flex min-h-screen flex-col items-center justify-center px-4 py-12">
      <div className="w-full max-w-lg">
        <div className="mb-8 text-center">
          <h1 className="mb-2 text-3xl font-bold tracking-tight text-foreground">
            MTG Price Scanner
          </h1>
          <p className="text-muted-foreground">
            Upload a picture of the card to get a price estimate
          </p>
        </div>

        <Card className="border-2 border-border bg-card p-6">
          <div className="mb-6">
            <h2 className="mb-1 text-sm font-semibold uppercase tracking-wide text-foreground">
              Step 1: Upload Image
            </h2>
            <p className="text-sm text-muted-foreground">
              Drag and drop or click to select a photo
            </p>
          </div>

          <div
            onDragOver={handleDragOver}
            onDragLeave={handleDragLeave}
            onDrop={handleDrop}
            onClick={handleUploadClick}
            className={`relative mb-6 flex min-h-[200px] cursor-pointer flex-col items-center justify-center rounded-lg border-2 border-dashed transition-colors ${
              isDragging
                ? "border-primary bg-primary/5"
                : "border-border hover:border-primary/50"
            } ${previewUrl ? "bg-muted/30" : "bg-background"}`}
          >
            {previewUrl ? (
              <div className="relative h-full w-full p-4">
                <img
                  src={previewUrl || "/placeholder.svg"}
                  alt="Preview"
                  className="mx-auto max-h-[180px] rounded-md object-contain"
                />
              </div>
            ) : (
              <div className="flex flex-col items-center gap-3 p-8 text-center">
                <div className="flex h-14 w-14 items-center justify-center rounded-full bg-muted">
                  <ImageIcon className="h-7 w-7 text-muted-foreground" />
                </div>
                <div>
                  <p className="font-medium text-foreground">
                    Drag & Drop a file here
                  </p>
                  <p className="text-sm text-muted-foreground">
                    or click to browse
                  </p>
                </div>
              </div>
            )}
            <input
              ref={fileInputRef}
              type="file"
              accept="image/*"
              onChange={handleFileChange}
              className="hidden"
            />
          </div>

          {selectedFile && (
            <div className="mb-6 flex items-center gap-2 rounded-md bg-muted px-3 py-2 text-sm">
              <Upload className="h-4 w-4 text-muted-foreground" />
              <span className="truncate text-foreground">{selectedFile.name}</span>
            </div>
          )}

          <Button
            onClick={onFindPrice}
            disabled={!selectedFile}
            className="w-full bg-primary text-primary-foreground hover:bg-primary/90"
            size="lg"
          >
            Find Price
          </Button>
        </Card>
      </div>
    </div>
  )
}
