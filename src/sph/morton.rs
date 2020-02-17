pub const MORTON_XBITS: u32 = 0b01010101_01010101_01010101_01010101;
pub const MORTON_YBITS: u32 = 0b10101010_10101010_10101010_10101010;

// Encodes two 16(!) bit numbers into a single 32bit morton code by interleaving the bits.
//
// Benchmark run 2020/02/16, i7-6700K
// According benchmark the lookup version is faster.
// Con: The benchmark warms its caches, so the lookup table is already loaded. encode_bitfiddle only uses a handful of constants.
//
// morton.encode/encode_bitfiddle
//                         time:   [2.7228 ns 2.7261 ns 2.7297 ns]
//                         change: [+0.1572% +0.2778% +0.4184%] (p = 0.00 < 0.05)
//                         Change within noise threshold.
// Found 10 outliers among 100 measurements (10.00%)
//   1 (1.00%) low severe
//   2 (2.00%) low mild
//   7 (7.00%) high mild
// morton.encode/encode_lookup
//                         time:   [1.3182 ns 1.3196 ns 1.3213 ns]
//                         change: [-0.1153% +0.0651% +0.2467%] (p = 0.50 > 0.05)
//                         No change in performance detected.
// Found 8 outliers among 100 measurements (8.00%)
//   6 (6.00%) high mild
//   2 (2.00%) high severe
pub use encode_lookup as encode;

// "Insert" a 0 bit after each of the 16 low bits of x
//
// via https://fgiesen.wordpress.com/2009/12/13/decoding-morton-codes/
// Example: input 0b1011_1100_1011_1101, output 0b01000101_01010000_01000101_01010001
#[inline]
fn part_1by1(x: u16) -> u32 {
    let mut x = x as u32; // x = ---- ---- ---- ---- fedc ba98 7654 3210
    x = (x ^ (x << 8)) & 0x00ff_00ff; // x = ---- ---- fedc ba98 ---- ---- 7654 3210
    x = (x ^ (x << 4)) & 0x0f0f_0f0f; // x = ---- fedc ---- ba98 ---- 7654 ---- 3210
    x = (x ^ (x << 2)) & 0x3333_3333; // x = --fe --dc --ba --98 --76 --54 --32 --10
    x = (x ^ (x << 1)) & 0x5555_5555; // x = -f-e -d-c -b-a -9-8 -7-6 -5-4 -3-2 -1-0
    x
}

// Encodes two 16(!) bit numbers into a single 32bit number by interleaving the bits.
#[inline]
pub fn encode_bitfiddle(x: u16, y: u16) -> u32 {
    (part_1by1(y) << 1) + part_1by1(x)
}

// Encodes two 16(!) bit numbers into a single 32bit morton code by interleaving the bits.
// Uses a byte lookup table.
//
// via
// https://graphics.stanford.edu/~seander/bithacks.html#InterleaveTableObvious
#[inline]
pub fn encode_lookup(x: u16, y: u16) -> u32 {
    const MORTON_TABLE256: [u16; 256] = [
        0x0000, 0x0001, 0x0004, 0x0005, 0x0010, 0x0011, 0x0014, 0x0015, 0x0040, 0x0041, 0x0044, 0x0045, 0x0050, 0x0051, 0x0054, 0x0055, 0x0100,
        0x0101, 0x0104, 0x0105, 0x0110, 0x0111, 0x0114, 0x0115, 0x0140, 0x0141, 0x0144, 0x0145, 0x0150, 0x0151, 0x0154, 0x0155, 0x0400, 0x0401,
        0x0404, 0x0405, 0x0410, 0x0411, 0x0414, 0x0415, 0x0440, 0x0441, 0x0444, 0x0445, 0x0450, 0x0451, 0x0454, 0x0455, 0x0500, 0x0501, 0x0504,
        0x0505, 0x0510, 0x0511, 0x0514, 0x0515, 0x0540, 0x0541, 0x0544, 0x0545, 0x0550, 0x0551, 0x0554, 0x0555, 0x1000, 0x1001, 0x1004, 0x1005,
        0x1010, 0x1011, 0x1014, 0x1015, 0x1040, 0x1041, 0x1044, 0x1045, 0x1050, 0x1051, 0x1054, 0x1055, 0x1100, 0x1101, 0x1104, 0x1105, 0x1110,
        0x1111, 0x1114, 0x1115, 0x1140, 0x1141, 0x1144, 0x1145, 0x1150, 0x1151, 0x1154, 0x1155, 0x1400, 0x1401, 0x1404, 0x1405, 0x1410, 0x1411,
        0x1414, 0x1415, 0x1440, 0x1441, 0x1444, 0x1445, 0x1450, 0x1451, 0x1454, 0x1455, 0x1500, 0x1501, 0x1504, 0x1505, 0x1510, 0x1511, 0x1514,
        0x1515, 0x1540, 0x1541, 0x1544, 0x1545, 0x1550, 0x1551, 0x1554, 0x1555, 0x4000, 0x4001, 0x4004, 0x4005, 0x4010, 0x4011, 0x4014, 0x4015,
        0x4040, 0x4041, 0x4044, 0x4045, 0x4050, 0x4051, 0x4054, 0x4055, 0x4100, 0x4101, 0x4104, 0x4105, 0x4110, 0x4111, 0x4114, 0x4115, 0x4140,
        0x4141, 0x4144, 0x4145, 0x4150, 0x4151, 0x4154, 0x4155, 0x4400, 0x4401, 0x4404, 0x4405, 0x4410, 0x4411, 0x4414, 0x4415, 0x4440, 0x4441,
        0x4444, 0x4445, 0x4450, 0x4451, 0x4454, 0x4455, 0x4500, 0x4501, 0x4504, 0x4505, 0x4510, 0x4511, 0x4514, 0x4515, 0x4540, 0x4541, 0x4544,
        0x4545, 0x4550, 0x4551, 0x4554, 0x4555, 0x5000, 0x5001, 0x5004, 0x5005, 0x5010, 0x5011, 0x5014, 0x5015, 0x5040, 0x5041, 0x5044, 0x5045,
        0x5050, 0x5051, 0x5054, 0x5055, 0x5100, 0x5101, 0x5104, 0x5105, 0x5110, 0x5111, 0x5114, 0x5115, 0x5140, 0x5141, 0x5144, 0x5145, 0x5150,
        0x5151, 0x5154, 0x5155, 0x5400, 0x5401, 0x5404, 0x5405, 0x5410, 0x5411, 0x5414, 0x5415, 0x5440, 0x5441, 0x5444, 0x5445, 0x5450, 0x5451,
        0x5454, 0x5455, 0x5500, 0x5501, 0x5504, 0x5505, 0x5510, 0x5511, 0x5514, 0x5515, 0x5540, 0x5541, 0x5544, 0x5545, 0x5550, 0x5551, 0x5554,
        0x5555,
    ];
    unsafe {
        (*MORTON_TABLE256.get_unchecked((y >> 8) as usize) as u32) << 17
            | (*MORTON_TABLE256.get_unchecked((x >> 8) as usize) as u32) << 16
            | (*MORTON_TABLE256.get_unchecked((y & 0xFF) as usize) as u32) << 1
            | (*MORTON_TABLE256.get_unchecked((x & 0xFF) as usize) as u32)
    }
}

// determines if a given morton code is within a rectangle given by morton codes\
#[allow(dead_code)]
pub(super) fn is_in_rect(m_cur: u32, min_morton: u32, max_morton: u32) -> bool {
    let max_morton_xbits = max_morton & MORTON_XBITS;
    let max_morton_ybits = max_morton & MORTON_YBITS;
    let min_morton_xbits = min_morton & MORTON_XBITS;
    let min_morton_ybits = min_morton & MORTON_YBITS;
    is_in_rect_presplit(m_cur, min_morton_xbits, min_morton_ybits, max_morton_xbits, max_morton_ybits)
}

// determines if a given morton code is within a rectangle given by pre-split morton codes
pub(super) fn is_in_rect_presplit(m_cur: u32, min_morton_xbits: u32, min_morton_ybits: u32, max_morton_xbits: u32, max_morton_ybits: u32) -> bool {
    let cur_x = m_cur & MORTON_XBITS;
    let cur_y = m_cur & MORTON_YBITS;

    cur_x >= min_morton_xbits && cur_y >= min_morton_ybits && cur_x <= max_morton_xbits && cur_y <= max_morton_ybits
}

// loads a bit pattern for a given dimension.
// Width of applied bit pattern is patternlen
// dim = 0 for x; dim = 1 for y
//
// Example:
//   bit pattern 1011 (patternlen==8) for dim=1=y is first spread out 1x0x_1x1x
//   then all these bits are replaced in value so if value was 1111_1111_0011_1111, it is now 1111_1111_1001_1111
fn load_bits(pattern: u32, patternlen: u32, value: u32, dim: u32) -> u32 {
    let wipe_mask = !(part_1by1(0xffff >> (16 - (patternlen / 2 + 1))) << dim); // clears affected bits
    let pattern = part_1by1(pattern as u16) << dim; // spreads pattern
    (value & wipe_mask) | pattern
}

// For a given morton index and a bounding rectangle in morton indices,
// finds the next index that is in the bounding rectangle.
//
// See decision table at the end of http://hermanntropf.de/media/multidimensionalrangequery.pdf
// This was tricky. Some more resources LITMAX/BIGMIN algorithm
// http://hermanntropf.de/media/multidimensionalrangequery.pdf
// https://web.archive.org/web/20180311015006/https://docs.raima.com/rdme/9_1/Content/GS/POIexample.htm
// https://stackoverflow.com/questions/30170783/how-to-use-morton-orderz-order-curve-in-range-search
pub fn find_bigmin(m_cur: u32, min_morton: u32, max_morton: u32) -> u32 {
    let mut min_morton = min_morton;
    let mut max_morton = max_morton;
    let mut bigmin = 0;
    for bitpos in (0..32_u32).rev() {
        let setbit = 1 << bitpos;
        let curbit = (m_cur & setbit) != 0;
        let minbit = (min_morton & setbit) != 0;
        let maxbit = (max_morton & setbit) != 0;
        let dim = bitpos % 2; // dim = 0 for x; dim = 1 for y
        let mask = 1 << (bitpos / 2);

        match (curbit, minbit, maxbit) {
            (false, false, false) => (),
            (false, false, true) => {
                bigmin = load_bits(mask, bitpos, min_morton, dim);
                max_morton = load_bits(mask - 1, bitpos, max_morton, dim);
            }
            (false, true, false) => unreachable!(),
            (false, true, true) => return min_morton,
            (true, false, false) => return bigmin,
            (true, false, true) => min_morton = load_bits(mask, bitpos, min_morton, dim),
            (true, true, false) => unreachable!(),
            (true, true, true) => (),
        }
    }
    bigmin
}

#[cfg(test)]
mod tests {
    mod encode {
        use super::super::*;

        #[test]
        fn lookup_works_for_examples() {
            assert_eq!(encode_lookup(2, 2), 12);
            assert_eq!(encode_lookup(3, 6), 45);
            assert_eq!(encode_lookup(4, 0), 16);
            assert_eq!(encode_lookup(0b1111_0001_0010_0000, 0b1001_1101_1000_1100), 0b1101_0111_1010_0011_1000_0100_1010_0000);
        }

        #[test]
        fn bitfiddle_works_for_examples() {
            assert_eq!(encode_bitfiddle(2, 2), 12);
            assert_eq!(encode_bitfiddle(3, 6), 45);
            assert_eq!(encode_bitfiddle(4, 0), 16);
            assert_eq!(encode_bitfiddle(0b1111_0001_0010_0000, 0b1001_1101_1000_1100), 0b1101_0111_1010_0011_1000_0100_1010_0000);
        }
    }

    mod find_bigmin {
        use super::super::*;

        #[test]
        fn jumps_to_next_pos_in_rect() {
            // from https://en.wikipedia.org/wiki/Z-order_curve#Use_with_one-dimensional_data_structures_for_range_searching
            assert_eq!(find_bigmin(16, 12, 45), 36);
            assert_eq!(find_bigmin(19, 12, 45), 36);
            assert_eq!(find_bigmin(29, 12, 45), 36);
            assert_eq!(find_bigmin(35, 12, 45), 36);
        }

        #[test]
        fn within_rect_gives_next_in_rect() {
            assert_eq!(find_bigmin(14, 12, 45), 15);
        }

        #[test]
        fn at_border_of_section_gives_next_in_rect() {
            assert_eq!(find_bigmin(15, 12, 45), 36);
        }
    }
}