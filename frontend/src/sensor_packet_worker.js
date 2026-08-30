self.onmessage = (event) => {
  const started = performance.now();
  try {
    const {
      header,
      width,
      height,
      rgbRgbaBuffer,
      depthMetersBuffer,
      depthClipM,
    } = event.data;
    const rgba = new Uint8Array(rgbRgbaBuffer);
    const depthMeters = new Float32Array(depthMetersBuffer);
    const rgb = new Uint8Array(width * height * 3);
    const depth = new Uint16Array(width * height);

    for (let targetY = 0; targetY < height; targetY += 1) {
      const sourceY = height - 1 - targetY;
      for (let x = 0; x < width; x += 1) {
        const source = (sourceY * width + x) * 4;
        const target = (targetY * width + x) * 3;
        rgb[target] = rgba[source];
        rgb[target + 1] = rgba[source + 1];
        rgb[target + 2] = rgba[source + 2];
        const depthValue = depthMeters[sourceY * width + x];
        depth[targetY * width + x] = Math.round(
          Math.min(depthClipM, Math.max(0, depthValue)) / depthClipM * 65535,
        );
      }
    }

    const depthBytes = new Uint8Array(depth.buffer);
    header.rgb_length = rgb.byteLength;
    header.depth_length = depthBytes.byteLength;
    const headerBytes = new TextEncoder().encode(JSON.stringify(header));
    const packet = new Uint8Array(
      8 + headerBytes.byteLength + rgb.byteLength + depthBytes.byteLength,
    );
    packet.set([85, 70, 87, 77], 0); // UFWM
    new DataView(packet.buffer).setUint32(4, headerBytes.byteLength, true);
    packet.set(headerBytes, 8);
    packet.set(rgb, 8 + headerBytes.byteLength);
    packet.set(depthBytes, 8 + headerBytes.byteLength + rgb.byteLength);
    self.postMessage(
      {
        type: 'packet',
        packet: packet.buffer,
        sequence: header.sequence,
        encode_ms: performance.now() - started,
      },
      [packet.buffer],
    );
  } catch (error) {
    self.postMessage({ type: 'error', error: String(error) });
  }
};
