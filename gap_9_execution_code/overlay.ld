SECTIONS
{   
    OVERLAY : NOCROSSREFS
    {
        .effvit_large
        {
            *(.effvit_large)
            *(.effvit_large.*)
        }

        .effvit_small
        {
            *(.effvit_small)
            *(.effvit_small.*)
        }
    } > L2_shared AT> overlay_area
}