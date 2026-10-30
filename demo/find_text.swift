// Finds a word in pictures with macOS's own text recognition, for record_claude_code.py.
//
//     swift find_text.swift WORD PICTURE...
//
// For each place a picture shows the word, in any case, it prints one line: the picture, then the box around the
// word in pixels from the top left corner (x, y, width, height). It prints nothing for a picture without the word.

import Foundation
import ImageIO
import Vision

let arguments = CommandLine.arguments
guard arguments.count >= 3 else {
    FileHandle.standardError.write("Use: swift find_text.swift WORD PICTURE...\n".data(using: .utf8)!)
    exit(2)
}
let word = arguments[1].lowercased()

for path in arguments.dropFirst(2) {
    guard let source = CGImageSourceCreateWithURL(URL(fileURLWithPath: path) as CFURL, nil),
          let image = CGImageSourceCreateImageAtIndex(source, 0, nil) else {
        FileHandle.standardError.write("Can't read \(path)\n".data(using: .utf8)!)
        exit(1)
    }
    let request = VNRecognizeTextRequest()
    request.recognitionLevel = .accurate
    request.usesLanguageCorrection = false
    try VNImageRequestHandler(cgImage: image, options: [:]).perform([request])
    let width = Double(image.width), height = Double(image.height)
    for observation in request.results ?? [] {
        guard let candidate = observation.topCandidates(1).first else { continue }
        let text = candidate.string
        var from = text.startIndex
        while let found = text.range(of: word, options: .caseInsensitive, range: from..<text.endIndex) {
            from = found.upperBound
            guard let box = try candidate.boundingBox(for: found)?.boundingBox else { continue }
            let x = box.minX * width, y = (1 - box.maxY) * height
            print(String(format: "%@\t%.0f\t%.0f\t%.0f\t%.0f", path, x, y, box.width * width, box.height * height))
        }
    }
}
