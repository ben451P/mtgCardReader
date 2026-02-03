"use client"

import React from "react"

import { ArrowLeft, DollarSign, Tag, Star, Package } from "lucide-react"
import { Button } from "@/components/ui/button"
import { Card } from "@/components/ui/card"

interface PriceStats {
  collector_num: string
  price: string
  price_foil: string
  rarity: string
}

interface ResultsPageProps {
  stats: PriceStats
  previewUrl: string | null
  onReset: () => void
}

export function ResultsPage({ stats, previewUrl, onReset }: ResultsPageProps) {
  return (
    <div className="flex min-h-screen flex-col items-center justify-center px-4 py-12">
      <div className="w-full max-w-lg">
        <div className="mb-8 text-center">
          <h1 className="mb-2 text-3xl font-bold tracking-tight text-foreground">
            Price Estimate (for the card)
          </h1>
          <p className="text-muted-foreground">
            Here&apos;s what we found
          </p>
        </div>

        <Card className="border-2 border-border bg-card p-6">
          {previewUrl && (
            <div className="mb-6 overflow-hidden rounded-lg bg-muted/30">
              <img
                src={previewUrl || "/placeholder.svg"}
                alt="Scanned item"
                className="mx-auto max-h-[160px] object-contain p-4"
              />
            </div>
          )}

          <div className="mb-6 rounded-lg border-2 border-primary bg-primary/5 p-4 text-center">
            <p className="mb-1 text-sm font-medium uppercase tracking-wide text-primary">
              Collector's Number
            </p>
            <p className="text-4xl font-bold text-foreground">
              {stats.collector_num}
            </p>
          </div>

          <div className="mb-6 grid grid-cols-2 gap-3">
            <StatCard
              icon={<Star className="h-4 w-4" />}
              label="Price"
              value={stats.price}
            />
            <StatCard
              icon={<Tag className="h-4 w-4" />}
              label="Foil Price"
              value={stats.price_foil}
            />
            <StatCard
              icon={<Package className="h-4 w-4" />}
              label="Card Rarity"
              value={stats.rarity}
            />
          </div>

          <Button
            onClick={onReset}
            variant="outline"
            className="w-full border-2 border-border bg-transparent text-foreground hover:bg-muted"
            size="lg"
          >
            <ArrowLeft className="mr-2 h-4 w-4" />
            Scan Another
          </Button>
        </Card>
      </div>
    </div>
  )
}

function StatCard({
  icon,
  label,
  value,
}: {
  icon: React.ReactNode
  label: string
  value: string
}) {
  return (
    <div className="rounded-lg border border-border bg-muted/30 p-3">
      <div className="mb-1 flex items-center gap-2 text-muted-foreground">
        {icon}
        <span className="text-xs font-medium uppercase tracking-wide">{label}</span>
      </div>
      <p className="font-semibold text-foreground">{value}</p>
    </div>
  )
}
