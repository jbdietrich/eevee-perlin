extern crate eevee_perlin;
extern crate image;

use std::fs::File;
use std::path::Path;
use eevee_perlin::PerlinNoiseGenerator;
use image::{Luma, ImageBuffer, ImageLuma8};

fn main() {
    let size = 200;
    let res = 40;
    let frames = 20;
    let frameres = 5;
    let space_range = size / res;
    let frame_range = frames / frameres;

    let mut png = PerlinNoiseGenerator::new(3, 4, &[space_range, space_range, frame_range]);
    for t in 0..frames {
        let mut imgbuf = ImageBuffer::new(size, size);
        for (x, y, pixel) in imgbuf.enumerate_pixels_mut() {
            let n = png.generate(vec![x as f64 / res as f64,
                                      y as f64 / res as f64,
                                      t as f64 / frameres as f64]);
            let chan = ((n + 1.0f64) / 2.0f64 * (255.0f64 + 0.5f64)) as u8;
            *pixel = Luma([chan]);
        }
        let ref mut fout = File::create(&Path::new(&format!("noiseframe{}.png", t))).unwrap();
        let _ = ImageLuma8(imgbuf).save(fout, image::PNG);
        println!("{}", t);
    }
}
