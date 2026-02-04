#pragma once

#include <gpu/common.h>

namespace cubvh {

// Triangle data structure
struct Triangle {

    __host__ __device__ Eigen::Vector3f sample_uniform_position(const Eigen::Vector2f& sample) const {
        float sqrt_x = std::sqrt(sample.x());
        float factor0 = 1.0f - sqrt_x;
        float factor1 = sqrt_x * (1.0f - sample.y());
        float factor2 = sqrt_x * sample.y();

        return factor0 * a + factor1 * b + factor2 * c;
    }

    __host__ __device__ float surface_area() const {
        return 0.5f * Eigen::Vector3f((b - a).cross(c - a)).norm();
    }

    __host__ __device__ Eigen::Vector3f normal() const {
        return (b - a).cross(c - a).normalized();
    }

    __host__ __device__ float ray_intersect(const Eigen::Vector3f &ro, const Eigen::Vector3f &rd, Eigen::Vector3f& n) const { // based on https://www.iquilezles.org/www/articles/intersectors/intersectors.htm
        Eigen::Vector3f v1v0 = b - a;
        Eigen::Vector3f v2v0 = c - a;
        Eigen::Vector3f rov0 = ro - a;
        n = v1v0.cross( v2v0 );
        Eigen::Vector3f q = rov0.cross( rd );
        float d = safe_divide(1.0f, rd.dot(n));
        float u = d*-q.dot( v2v0 );
        float v = d* q.dot( v1v0 );
        float t = d*-n.dot( rov0 );
        if( u<0.0f || u>1.0f || v<0.0f || (u+v)>1.0f || t<0.0f) t = 1e6f;
        return t; // Eigen::Vector3f( t, u, v );
    }

    __host__ __device__ float ray_intersect(const Eigen::Vector3f &ro, const Eigen::Vector3f &rd) const {
        Eigen::Vector3f n;
        return ray_intersect(ro, rd, n);
    }

    __host__ __device__ float distance_sq(const Eigen::Vector3f& pos) const {
        Eigen::Vector3f v21 = b - a; Eigen::Vector3f p1 = pos - a;
        Eigen::Vector3f v32 = c - b; Eigen::Vector3f p2 = pos - b;
        Eigen::Vector3f v13 = a - c; Eigen::Vector3f p3 = pos - c;
        Eigen::Vector3f nor = v21.cross(v13);
        float nor_sq = nor.squaredNorm();
        bool is_degenerate = nor_sq < 1e-24f;
        bool is_inside =
            (v21.cross(nor).dot(p1)) >= 0.0f &&
            (v32.cross(nor).dot(p2)) >= 0.0f &&
            (v13.cross(nor).dot(p3)) >= 0.0f;

        if (!is_inside || is_degenerate) {
            // 3 edges
            return std::min(
                std::min(
                    (v21 * clamp(v21.dot(p1) / v21.squaredNorm(), 0.0f, 1.0f)-p1).squaredNorm(),
                    (v32 * clamp(v32.dot(p2) / v32.squaredNorm(), 0.0f, 1.0f)-p2).squaredNorm()
                ),
                (v13 * clamp(v13.dot(p3) / v13.squaredNorm(), 0.0f, 1.0f)-p3).squaredNorm()
            );
        }
        else {
            // 1 face
            return nor.dot(p1)*nor.dot(p1)/nor_sq;
        }
    }

    __host__ __device__ float distance(const Eigen::Vector3f& pos) const {
        return std::sqrt(distance_sq(pos));
    }

    __host__ __device__ Eigen::Vector3f closest_point_to_segment(const Eigen::Vector3f& p, const Eigen::Vector3f& a, const Eigen::Vector3f& b) const {
        Eigen::Vector3f ab = b - a;
        float len_sq = ab.squaredNorm();

        if (len_sq < 1e-12f) {
            return a; 
        }
        float t = (p - a).dot(ab) / len_sq;
        
        if (t < 0.0f) t = 0.0f;
        else if (t > 1.0f) t = 1.0f;

        return a + t * ab;
    }

    __host__ __device__ Eigen::Vector3f closest_point(Eigen::Vector3f point) const {
        Eigen::Vector3f v21 = b - a; Eigen::Vector3f p1 = point - a;
        Eigen::Vector3f v32 = c - b; Eigen::Vector3f p2 = point - b;
        Eigen::Vector3f v13 = a - c; Eigen::Vector3f p3 = point - c;
        Eigen::Vector3f nor = v21.cross(v13);
        float nor_sq = nor.squaredNorm();
        bool is_degenerate = nor_sq < 1e-24f;
        bool is_inside =
            (v21.cross(nor).dot(p1)) >= 0.0f &&
            (v32.cross(nor).dot(p2)) >= 0.0f &&
            (v13.cross(nor).dot(p3)) >= 0.0f;

        if (!is_inside || is_degenerate) {
            // 3 edges
            Eigen::Vector3f c1 = closest_point_to_segment(point, a, b);
            Eigen::Vector3f c2 = closest_point_to_segment(point, b, c);
            Eigen::Vector3f c3 = closest_point_to_segment(point, c, a);

            float d1 = (point - c1).squaredNorm();
            float d2 = (point - c2).squaredNorm();
            float d3 = (point - c3).squaredNorm();

            if (d1 <= d2 && d1 <= d3) return c1;
            if (d2 <= d3) return c2;
            return c3;
        }
        else {
            // 1 face
            float dist_factor = nor.dot(point - a) / nor_sq;
            return point - nor * dist_factor;
        }
    }

    __host__ __device__ Eigen::Vector3f barycentric(const Eigen::Vector3f& p) const {
        Eigen::Vector3f v0 = b - a;
        Eigen::Vector3f v1 = c - a;
        Eigen::Vector3f v2 = p - a;

        float d00 = v0.dot(v0);
        float d01 = v0.dot(v1);
        float d11 = v1.dot(v1);
        float d20 = v2.dot(v0);
        float d21 = v2.dot(v1);

        float denom = d00 * d11 - d01 * d01;
        float v = (d11 * d20 - d01 * d21) / denom;
        float w = (d00 * d21 - d01 * d20) / denom;
        float u = 1.0 - v - w;
        
        return Eigen::Vector3f(u, v, w);
    }

    __host__ __device__ Eigen::Vector3f centroid() const {
        return (a + b + c) / 3.0f;
    }

    __host__ __device__ float centroid(int axis) const {
        return (a[axis] + b[axis] + c[axis]) / 3;
    }

    __host__ __device__ void get_vertices(Eigen::Vector3f v[3]) const {
        v[0] = a;
        v[1] = b;
        v[2] = c;
    }

    Eigen::Vector3f a, b, c;
    int64_t id;
};


inline std::ostream& operator<<(std::ostream& os, const Triangle& triangle) {
    os << "[";
    os << "a=[" << triangle.a.x() << "," << triangle.a.y() << "," << triangle.a.z() << "], ";
    os << "b=[" << triangle.b.x() << "," << triangle.b.y() << "," << triangle.b.z() << "], ";
    os << "c=[" << triangle.c.x() << "," << triangle.c.y() << "," << triangle.c.z() << "]";
    os << "]";
    return os;
}


}