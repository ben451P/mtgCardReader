"use client"

import { Loader2 } from "lucide-react"

export function LoadingPage() {
  return (
    <div className="flex min-h-screen flex-col items-center justify-center px-4">
      <div className="flex flex-col items-center gap-6 text-center">
        <div className="relative">
          <div className="h-20 w-20 rounded-full border-4 border-muted" />
          <div className="absolute inset-0 flex items-center justify-center">
            <Loader2 className="h-10 w-10 animate-spin text-primary" />
          </div>
        </div>
        <div>
          <h2 className="mb-2 text-2xl font-bold text-foreground">
            Analyzing Image
          </h2>
          <p className="text-muted-foreground">
            Please wait while we find the best price estimate...
          </p>
        </div>
        <div className="flex gap-1">
          <span className="h-2 w-2 animate-bounce rounded-full bg-primary [animation-delay:0ms]" />
          <span className="h-2 w-2 animate-bounce rounded-full bg-primary [animation-delay:150ms]" />
          <span className="h-2 w-2 animate-bounce rounded-full bg-primary [animation-delay:300ms]" />
        </div>
      </div>
    </div>
  )
}
