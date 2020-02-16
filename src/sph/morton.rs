// via https://fgiesen.wordpress.com/2009/12/13/decoding-morton-codes/

// "Insert" a 0 bit after each of the 16 low bits of x
// Example: input 0b1011_1100_1011_1101, output 0b01000101_01010000_01000101_01010001
pub(super) fn part_1by1(mut x: u32) -> u32 {
    x &= 0x0000_ffff; // x = ---- ---- ---- ---- fedc ba98 7654 3210
    x = (x ^ (x << 8)) & 0x00ff_00ff; // x = ---- ---- fedc ba98 ---- ---- 7654 3210
    x = (x ^ (x << 4)) & 0x0f0f_0f0f; // x = ---- fedc ---- ba98 ---- 7654 ---- 3210
    x = (x ^ (x << 2)) & 0x3333_3333; // x = --fe --dc --ba --98 --76 --54 --32 --10
    x = (x ^ (x << 1)) & 0x5555_5555; // x = -f-e -d-c -b-a -9-8 -7-6 -5-4 -3-2 -1-0
    x
}

// encodes two 16(!) bit numbers into a single 32bit number by interleaving the bits.
pub(super) fn encode(x: u32, y: u32) -> u32 {
    (part_1by1(y) << 1) + part_1by1(x)
}

// loads a bit pattern for a given dimension.
// Width of applied bit pattern is patternlen
// dim = 0 for x; dim = 1 for y
//
// Example:
//   bit pattern 1011 (patternlen==8) for dim=1=y is first spread out 1x0x_1x1x
//   then all these bits are replaced in value so if value was 1111_1111_0011_1111, it is now 1111_1111_1001_1111
fn load_bits(pattern: u32, patternlen: u32, value: u32, dim: u32) -> u32 {
    let wipe_mask = !(super::morton::part_1by1(0xffff >> (16 - (patternlen / 2 + 1))) << dim); // clears affected bits
    let pattern = super::morton::part_1by1(pattern) << dim; // spreads pattern
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
pub(super) fn find_bigmin(m_cur: u32, m_min: u32, m_max: u32) -> u32 {
    let mut m_min = m_min;
    let mut m_max = m_max;
    let mut bigmin = 0;
    for bitpos in (0..32_u32).rev() {
        let setbit = 1 << bitpos;
        let minbit = (m_min & setbit) != 0;
        let maxbit = (m_max & setbit) != 0;
        let curbit = (m_cur & setbit) != 0;
        let dim = bitpos % 2; // dim = 0 for x; dim = 1 for y
        let mask = 1 << (bitpos / 2);

        if curbit {
            if !minbit && !maxbit {
                return bigmin;
            } else if !minbit && maxbit {
                m_min = load_bits(mask, bitpos, m_min, dim);
            } else if minbit && !maxbit {
                unreachable!();
            }
        } else {
            if !minbit && maxbit {
                bigmin = load_bits(mask, bitpos, m_min, dim);
                m_max = load_bits(mask - 1, bitpos, m_max, dim);
            } else if minbit && !maxbit {
                unreachable!();
            } else if minbit && maxbit {
                return m_min;
            }
        }
    }
    bigmin
}

// Don't know if this is one faster. Need to benchmark. Might be also case dependent
// Assembly looks promising, but above bit twiddle method is also different depending on targe processor (compiler uses fancy instructions if told that it is allowed to)
// See https://rust.godbolt.org/z/Aijui7

// via
// https://graphics.stanford.edu/~seander/bithacks.html#InterleaveTableObvious

#[allow(dead_code)]
pub fn encode_lookup(x: u32, y: u32) -> u32 {
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
