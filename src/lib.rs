use std::collections::HashMap;

type Point = f64;
type Coord = i64;

#[derive(Debug)]
pub struct PointValueError;

pub struct PerlinNoiseGenerator<'a> {
    dimension: usize,
    octaves: i32,
    scale_factor: f64,
    gradient_cache: HashMap<Vec<Coord>, Gradients>,
    tiles: Vec<u32>,
    rng: &'a Fn() -> f64,
}

#[derive(Debug)]
struct Gradients(Vec<Point>);

impl Gradients {
    fn from_dimension(dimension: usize, rng: &Fn() -> f64) -> Gradients {
        let mut gradients = Gradients(Vec::with_capacity(dimension));

        if dimension == 1 {
            gradients.0.push(rng());
            gradients
        } else {
            let random_points =
                (0..dimension).map(|_| rng()).collect::<Vec<Point>>();
            let scale = random_points.iter().map(|x| x * x).sum::<Point>().powf(-0.5);
            random_points.iter().fold(gradients, |mut memo, point| {
                memo.0.push(point * scale);
                memo
            })
        }
    }
}

#[derive(Debug)]
struct Bounds {
    min: Coord,
    max: Coord,
}

impl Bounds {
    fn cartesian_product(&self, others: &[Bounds]) -> Vec<Vec<Coord>> {
        let product = vec![vec![self.min], vec![self.max]];

        others.iter().fold(product, |memo, other| {
            memo.iter()
                .flat_map(|partial| {
                    let mut with_min = partial.clone();
                    let mut with_max = partial.clone();

                    with_min.push(other.min);
                    with_max.push(other.max);

                    vec![with_min, with_max]
                })
                .collect()
        })
    }
}

impl<'a> PerlinNoiseGenerator<'a> {
    pub fn new(dimension: usize, octaves: i32, tile: &[u32], rng: &'a Fn() -> f64) -> PerlinNoiseGenerator<'a> {
        let mut tiles = Vec::with_capacity(tile.len() + dimension);
        tiles.extend_from_slice(tile);
        for _ in 0..dimension {
            tiles.push(0);
        }

        PerlinNoiseGenerator {
            dimension: dimension,
            octaves: octaves,
            scale_factor: 2.0_f64 * (dimension as f64).powf(-0.5),
            gradient_cache: HashMap::new(),
            tiles: tiles,
            rng: rng,
        }
    }

    fn get_plain_noise(&mut self, points: Vec<Point>) -> Result<Point, PointValueError> {
        if points.len() != self.dimension {
            return Err(PointValueError);
        }

        let grid_bounds = points.iter()
            .map(|point| {
                let min_coord = point.floor() as i64;
                Bounds {
                    min: min_coord,
                    max: min_coord + 1,
                }
            })
            .collect::<Vec<Bounds>>();

        let mut product = grid_bounds[0].cartesian_product(&grid_bounds[1..]);
        let mut dots = Vec::with_capacity(product.len());

        for coords in product.drain(..) {
            let gradients = self.gradient_cache
                .entry(coords.clone())
                .or_insert(Gradients::from_dimension(self.dimension, self.rng));

            let dot = (0..self.dimension).fold(0.0_f64, |memo, i| {
                memo + gradients.0[i] * (points[i] - coords[i] as Point)
            });
            dots.push(dot);
        }

        let mut dim = self.dimension;
        while dots.len() > 1 {
            dim -= 1;
            let s = smoothstep(points[dim] - grid_bounds[dim].min as Point);

            let mut next_dots = vec![];
            while !dots.is_empty() {
                next_dots.push(lerp(s, dots.remove(0), dots.remove(0)));
            }
            dots = next_dots;
        }
        Ok(dots[0] * self.scale_factor)
    }

    pub fn generate(&mut self, points: Vec<Point>) -> Point {
        let ret = (0..self.octaves).fold(0.0_f64, |memo, octave| {
            let scaled_octave = 2.0_f64.powi(octave);
            let new_points = points.iter()
                .enumerate()
                .map(|(i, point)| {
                    let mut new_point = point.clone();
                    new_point *= scaled_octave;
                    if self.tiles[i] != 0 {
                        new_point %= (self.tiles[i] as f64) * scaled_octave;
                    }
                    new_point
                })
                .collect();
            memo + self.get_plain_noise(new_points).unwrap() / scaled_octave
        });
        ret / 2.0_f64 - 2.0_f64.powi(1 - self.octaves)
    }
}

fn smoothstep(point: Point) -> Point {
    point * point * (3. - 2. * point)
}

fn lerp(t: Point, a: Point, b: Point) -> Point {
    t.mul_add((b - a), a)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn it_does_not_return_nan() {
        let mut png = PerlinNoiseGenerator::new(3, 4, &[2, 2, 3]);
        assert!(!png.generate(vec![10.10, 10.9, 10.0]).is_nan());
    }
}
