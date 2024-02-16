use rand::rngs::ThreadRng;
use rand::Rng;
use rayon::prelude::*;
use std::fs::File;
use std::io::{BufWriter, Write};
use std::sync::Arc;
use std::env;

use indicatif::ProgressBar;

const PI: f64 = 3.14159265358979323846;

#[derive(Clone, Copy, Debug)]
struct Vec {
    x: f64,
    y: f64,
    z: f64,
}

impl Vec {
    fn new(x: f64, y: f64, z: f64) -> Vec {
        Vec { x, y, z }
    }

    fn operator_add(&self, other: Vec) -> Vec {
        Vec::new(self.x + other.x, self.y + other.y, self.z + other.z)
    }

    fn operator_sub(&self, other: Vec) -> Vec {
        Vec::new(self.x - other.x, self.y - other.y, self.z - other.z)
    }

    fn operator_mul(&self, other: Vec) -> Vec {
        Vec::new(self.x * other.x, self.y * other.y, self.z * other.z)
    }

    fn operator_times(&self, t: f64) -> Vec {
        Vec::new(self.x * t, self.y * t, self.z * t)
    }

    fn dot(&self, other: Vec) -> f64 {
        self.x * other.x + self.y * other.y + self.z * other.z
    }

    fn cross(&self, other: Vec) -> Vec {
        Vec::new(
            self.y * other.z - self.z * other.y,
            self.z * other.x - self.x * other.z,
            self.x * other.y - self.y * other.x,
        )
    }

    fn norm(&self) -> Vec {
        let len = (self.x * self.x + self.y * self.y + self.z * self.z).sqrt();
        self.operator_times(1.0 / len)
    }
}

#[derive(Debug)]
struct Ray {
    o: Vec, // Origin
    d: Vec, // Direction
}

impl Ray {
    fn new(o: Vec, d: Vec) -> Ray {
        Ray { o, d }
    }
}

#[derive(Clone, Copy, Debug)]
enum ReflType {
    Diffuse,
    Specular,
    Refractive,
}

#[derive(Clone, Copy, Debug)]
struct Sphere {
    rad: f64,   // Radius
    p: Vec,     // Position
    e: Vec,     // Emission
    c: Vec,     // Color
    refl: ReflType, // Reflection type (DIFFuse, SPECular, REFRactive)
}

impl Sphere {
    fn new(rad: f64, p: Vec, e: Vec, c: Vec, refl: ReflType) -> Sphere {
        Sphere { rad, p, e, c, refl }
    }

    fn intersect(&self, ray: &Ray) -> Option<f64> {
        let op = self.p.operator_sub(ray.o);
        let eps = 1e-4;
        let b = op.dot(ray.d);
        let det = b*b - op.dot(op) + self.rad*self.rad;
        if det < 0.0 { return None }
        else {
            let det_sqrt = det.sqrt();
            let t1 = b - det_sqrt;
            let t2 = b + det_sqrt;
            if t1 > eps { Some(t1) }
            else if t2 > eps { Some(t2) }
            else { None }
        }
    }
}

fn clamp(x: f64) -> f64 {
    if x < 0.0 { 0.0 } else if x > 1.0 { 1.0 } else { x }
}

fn to_int(x: f64) -> i32 {
    (clamp(x).powf(1.0/2.2) * 255.0 + 0.5) as i32
}

fn intersect(ray: &Ray, t: &mut f64, id: &mut usize, spheres: &[Sphere]) -> bool {
    let mut hit = false;
    let inf = 1e20;
    *t = inf;
    for (i, sphere) in spheres.iter().enumerate() {
        if let Some(d) = sphere.intersect(ray) {
            if d < *t {
                *t = d;
                *id = i;
                hit = true;
            }
        }
    }
    hit
}

fn radiance(ray: &Ray, depth: i32, rng: &mut ThreadRng, spheres: &[Sphere]) -> Vec {
    let mut t = 0.0; // distance to intersection
    let mut id = 0; // id of intersected object
    if !intersect(ray, &mut t, &mut id, spheres) {
        return Vec::new(0.0, 0.0, 0.0); // if miss, return black
    }
    let obj = spheres[id]; // the hit object

    let x = ray.o.operator_add(ray.d.operator_times(t));
    let n = x.operator_sub(obj.p).norm();
    let nl = if n.dot(ray.d) < 0.0 {n} else {n.operator_times(-1.0)};
    let mut f = obj.c;

    let p = if f.x > f.y && f.x > f.z {f.x} else if f.y > f.z {f.y} else {f.z}; // max reflection
    let depth = depth + 1;
    if depth > 5 {
        if rng.gen::<f64>() < p {
            f = f.operator_times(1.0 / p);
        } else {
            return obj.e; // R.R.
        }
    }

    match obj.refl {
        ReflType::Diffuse => {
            let r1 = 2.0 * PI * rng.gen::<f64>();
            let r2 = rng.gen::<f64>();
            let r2s = r2.sqrt();

            let w = nl;
            let u = if w.x.abs() > 0.1 { Vec::new(0.0, 1.0, 0.0) } else { Vec::new(1.0, 0.0, 0.0) }.cross(w).norm();
            let v = w.cross(u);
            let d = u.operator_times(r1.cos() * r2s).operator_add(v.operator_times(r1.sin() * r2s).operator_add(w.operator_times((1.0 - r2).sqrt()))).norm();
            obj.e.operator_add(f.operator_mul(radiance(&Ray::new(x, d), depth, rng, spheres)))
        },
        ReflType::Specular => {
            obj.e.operator_add(f.operator_mul(radiance(&Ray::new(x, ray.d.operator_sub(n.operator_times(2.0 * n.dot(ray.d)))), depth, rng, spheres)))
        },
        ReflType::Refractive => {
            let refl_ray = Ray::new(x, ray.d.operator_sub(n.operator_times(2.0 * n.dot(ray.d))));
            let into = n.dot(nl) > 0.0; // Ray from outside going into the medium?
            let nc = 1.0;
            let nt = 1.5;
            let nnt = if into {nc / nt} else {nt / nc};
            let ddn = ray.d.dot(nl);
            let cos2t = 1.0 - nnt * nnt * (1.0 - ddn * ddn);

            if cos2t < 0.0 { // Total internal reflection
                return obj.e.operator_add(f.operator_mul(radiance(&refl_ray, depth, rng, spheres)));
            }

            let tdir = (ray.d.operator_times(nnt).operator_sub(n.operator_times(((into as i32 as f64)*2.0-1.0) * (ddn * nnt + cos2t.sqrt())))).norm();
            let a = nt - nc;
            let b = nt + nc;
            let r0 = a * a / (b * b);
            let c = 1.0 - (if into { -ddn } else { tdir.dot(n) });
            let re = r0 + (1.0 - r0) * c.powi(5);
            let tr = 1.0 - re;
            let p = 0.25 + 0.5 * re;
            let rp = re / p;
            let tp = tr / (1.0 - p);

            let rand: f64 = rng.gen();
            if depth > 2 {
                if rand < p { // Reflect
                    obj.e.operator_add(f.operator_mul(radiance(&refl_ray, depth, rng, spheres).operator_times(rp)))
                } else { // Refract
                    obj.e.operator_add(f.operator_mul(radiance(&Ray::new(x, tdir), depth, rng, spheres).operator_times(tp)))
                }
            } else { // Reflect and refract
                obj.e.operator_add(f.operator_mul(radiance(&refl_ray, depth, rng, spheres).operator_times(re).operator_add(radiance(&Ray::new(x, tdir), depth, rng, spheres).operator_times(tr))))
            }
        },
    }
}

fn main() {
    let args: std::vec::Vec<String> = env::args().collect();

    

    let width = 1024;
    let height = 768;
    let samples = if args.len() > 1 {
        let first_arg = &args[1];
        first_arg.parse::<i32>().unwrap()
    } else {
        40
    }; // Increase for better quality

    let pb = ProgressBar::new(samples as u64 * width as u64 * height as u64 * 4);

    let cam = Ray::new(Vec::new(50.0, 52.0, 295.6), Vec::new(0.0, -0.042612, -1.0).norm());
    let cx = Vec::new(width as f64 * 0.5135 / height as f64, 0.0, 0.0);
    let cy = (cx.cross(cam.d)).norm().operator_times(0.5135);
    // let mut rng = rand::thread_rng();

    let spheres = vec![ 
        // Scene:   radius,         position,                           emission,                   color,                                          material
        Sphere::new(1e5,   Vec::new(1e5 + 1.0, 40.8, 81.6),    Vec::new(0.0, 0.0, 0.0),    Vec::new(0.75, 0.25, 0.25),                    ReflType::Diffuse),    // Left
        Sphere::new(1e5,   Vec::new(-1e5 + 99.0, 40.8, 81.6),  Vec::new(0.0, 0.0, 0.0),    Vec::new(0.25, 0.25, 0.75),                    ReflType::Diffuse),    // Right
        Sphere::new(1e5,   Vec::new(50.0, 40.8, 1e5),          Vec::new(0.0, 0.0, 0.0),    Vec::new(0.75, 0.75, 0.75),                    ReflType::Diffuse),    // Back
        Sphere::new(1e5,   Vec::new(50.0, 40.8, -1e5 + 170.0), Vec::new(0.0, 0.0, 0.0),    Vec::new(0.0, 0.0, 0.0),                       ReflType::Diffuse),    // Front
        Sphere::new(1e5,   Vec::new(50.0, 1e5, 81.6),          Vec::new(0.0, 0.0, 0.0),    Vec::new(0.75, 0.75, 0.75),                    ReflType::Diffuse),    // Bottom
        Sphere::new(1e5,   Vec::new(50.0, -1e5 + 82.0, 81.6),  Vec::new(0.0, 0.0, 0.0),    Vec::new(0.75, 0.75, 0.75),                    ReflType::Diffuse),    // Top
        Sphere::new(16.5,  Vec::new(27.0, 16.5, 47.0),         Vec::new(0.0, 0.0, 0.0),    Vec::new(1.0, 1.0, 1.0).operator_times(0.999), ReflType::Specular),   // Mirror
        Sphere::new(16.5,  Vec::new(73.0, 16.5, 78.0),         Vec::new(0.0, 0.0, 0.0),    Vec::new(1.0, 1.0, 1.0).operator_times(0.999), ReflType::Refractive), // Glass
        Sphere::new(600.0, Vec::new(50.0, 681.6 - 0.27, 81.6), Vec::new(12.0, 12.0, 12.0), Vec::new(0.0, 0.0, 0.0),                       ReflType::Diffuse)     // Light
    ];
    let spheres = Arc::new(spheres);

    let mut img = vec![Vec::new(0.0, 0.0, 0.0); width * height];
    img.par_iter_mut().enumerate().for_each(|(i, pixel)| {
        let x = i % width;
        let y = height - i / width - 1;
        let mut rng = rand::thread_rng();

        for sy in 0..2 { // 2x2 subpixel rows
            for sx in 0..2 { // 2x2 subpixel cols
                let mut r = Vec::new(0.0, 0.0, 0.0);
                for _ in 0..samples {
                    let r1 = 2.0 * rng.gen::<f64>();
                    let dx = if r1 < 1.0 { r1.sqrt() - 1.0 } else { 1.0 - (2.0 - r1).sqrt() };
                    let r2 = 2.0 * rng.gen::<f64>();
                    let dy = if r2 < 1.0 { r2.sqrt() - 1.0 } else { 1.0 - (2.0 - r2).sqrt() };
                    let d = cx.operator_times(((sx as f64 + 0.5 + dx) / 2.0 + x as f64) / width as f64 - 0.5)
                        .operator_add(cy.operator_times(((sy as f64 + 0.5 + dy) / 2.0 + y as f64) / height as f64 - 0.5))
                        .operator_add(cam.d);
                    r = r.operator_add(radiance(&Ray::new(cam.o.operator_add(d.operator_times(140.0)), d.norm()), 0, &mut rng, spheres.as_ref()).operator_times(1.0 / samples as f64));
                    pb.inc(1);
                }
                *pixel = pixel.operator_add(Vec::new(clamp(r.x), clamp(r.y), clamp(r.z)).operator_times(0.25));
            }
        }
    });

    let mut file = BufWriter::new(File::create("image.ppm").unwrap());
    writeln!(file, "P3\n{} {}\n{}", width, height, 255).unwrap();
    for pixel in img {
        writeln!(file, "{} {} {}", to_int(pixel.x), to_int(pixel.y), to_int(pixel.z)).unwrap();
    }
    println!("Finished.");
}