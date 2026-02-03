import { NextRequest, NextResponse } from "next/server"

export async function POST(request: NextRequest) {
  try {
    // Get the form data with the image
    const formData = await request.formData()
    const image = formData.get("image") as File | null

    if (!image) {
      return NextResponse.json(
        { error: "No image provided" },
        { status: 400 }
      )
    }

    // Simulate processing delay (2-3 seconds)

    const flaskResponse = await fetch("http://localhost:5000/api/fetch_price", {
      method: "POST",
      body: formData,
    })
    return NextResponse.json(await flaskResponse.json())

  } catch (error) {
    console.error("[v0] Error processing image:", error)
    return NextResponse.json(
      { error: "Failed to process image" },
      { status: 500 }
    )
  }
}
