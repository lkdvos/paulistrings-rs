use super::*;

#[test]
fn tags_pack_kind_and_epoch_below_the_guaranteed_tag_bound() {
    assert_eq!(MpiTransport::tag(KIND_EXCHANGE_HEADER, 0), 0);
    assert_eq!(MpiTransport::tag(KIND_EXCHANGE_PART, 0), 1);
    assert_eq!(MpiTransport::tag(KIND_GATHER_PART, 0), 3);
    assert_eq!(MpiTransport::tag(KIND_EXCHANGE_HEADER, 1), 16);
    assert_eq!(MpiTransport::tag(KIND_EXCHANGE_PART, 1), 17);
    // The last epoch before the wrap sits exactly on 32767.
    assert_eq!(MpiTransport::tag(KIND_GATHER_PART + 12, EPOCHS - 1), 32767);
    for epoch in 0..EPOCHS {
        for kind in [
            KIND_EXCHANGE_HEADER,
            KIND_EXCHANGE_PART,
            KIND_GATHER_HEADER,
            KIND_GATHER_PART,
        ] {
            let tag = MpiTransport::tag(kind, epoch);
            assert!((0..=32767).contains(&tag), "epoch {epoch} kind {kind}");
            assert_eq!(tag & 0xf, kind);
        }
    }
    assert_eq!(
        MpiTransport::tag(KIND_EXCHANGE_PART, EPOCHS),
        MpiTransport::tag(KIND_EXCHANGE_PART, 0),
    );
}

#[test]
fn header_round_trips_through_its_byte_form() {
    let a = [1u8, 2, 3];
    let b: [u8; 0] = [];
    let c = [7u8; 40];
    let parts: Vec<&[u8]> = vec![&a, &b, &c];

    let words = encode_header(3, &parts);
    assert_eq!(words[0] & 0xffff_ffff, u64::from(WIRE_VERSION));
    assert_eq!(words[0] >> 32, 3);
    assert_eq!(words[1], 3);
    assert_eq!(&words[2..], &[3, 0, 40]);

    let bytes: &[u8] = bytemuck::cast_slice(&words);
    assert_eq!(decode_header(bytes, 3, 0), vec![3, 0, 40]);
}

#[test]
fn an_empty_payload_encodes_as_a_two_word_header() {
    let words = encode_header(0, &[]);
    assert_eq!(words.len(), 2);
    let bytes: &[u8] = bytemuck::cast_slice(&words);
    assert_eq!(decode_header(bytes, 0, 0), Vec::<usize>::new());
}

#[test]
#[should_panic(expected = "wire version")]
fn a_header_from_another_wire_version_is_rejected() {
    let mut words = encode_header(1, &[]);
    words[0] = (words[0] & !0xffff_ffff) | 99;
    let bytes: &[u8] = bytemuck::cast_slice(&words);
    let _ = decode_header(bytes, 1, 0);
}

#[test]
#[should_panic(expected = "not the expected rank")]
fn a_header_from_the_wrong_rank_is_rejected() {
    let words = encode_header(2, &[]);
    let bytes: &[u8] = bytemuck::cast_slice(&words);
    let _ = decode_header(bytes, 5, 0);
}

#[test]
#[should_panic(expected = "declared")]
fn a_header_whose_part_count_disagrees_with_its_length_is_rejected() {
    let a = [1u8, 2];
    let mut words = encode_header(0, &[&a]);
    words[1] = 4;
    let bytes: &[u8] = bytemuck::cast_slice(&words);
    let _ = decode_header(bytes, 0, 0);
}

#[test]
#[should_panic(expected = "at least two")]
fn a_truncated_header_is_rejected() {
    let _ = decode_header(&[0u8; 8], 0, 0);
}

#[test]
fn chunk_counts_agree_with_slice_chunks_at_the_boundaries() {
    let buf = vec![0u8; 4096];
    for chunk in [1usize, 2, 3, 1024, 4095, 4096, 4097] {
        for len in [
            0,
            1,
            chunk.saturating_sub(1),
            chunk,
            chunk + 1,
            2 * chunk,
            2 * chunk + 1,
        ] {
            if len > buf.len() {
                continue;
            }
            assert_eq!(
                chunk_count(len, chunk),
                buf[..len].chunks(chunk).count(),
                "len {len} chunk {chunk}",
            );
        }
    }
    assert_eq!(chunk_count(0, 1 << 30), 0);
    assert_eq!(chunk_count(1 << 30, 1 << 30), 1);
    assert_eq!(chunk_count((1 << 30) + 1, 1 << 30), 2);
}

#[test]
fn version_matching_finds_the_build_version_in_the_runtime_banner() {
    assert!(version_matches(
        "Open MPI 5.0.6 (Language: C)",
        "Open MPI v5.0.6, package: Open MPI build, ident: 5.0.6",
    ));
    assert!(!version_matches(
        "Open MPI 5.0.6 (Language: C)",
        "Open MPI v4.1.6, package: Open MPI build, ident: 4.1.6",
    ));
    // No number to compare: stay quiet.
    assert!(version_matches("unknown", "Open MPI v5.0.6"));
}
