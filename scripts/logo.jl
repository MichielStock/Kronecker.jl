using Luxor
import Random
Random.seed!(123);

function draw_kronecker(A, B)

    K = kron(A, B)'

    n, m = size(K)

    tiles = Tiler(12.5n, 12.5m, m, n, margin=0)

    for ((pos, n), k) in zip(tiles, K)
        if k==1
            sethue(Luxor.darker_green)
        elseif k==2
            sethue(Luxor.darker_red)
        elseif k==3
            sethue(Luxor.darker_purple)
        else
            randomhue()
        end
        if k > 0
            circle(pos, 5, :fill)
        end
    end
end

A = [1 1 0 0;
    0 1 0 1;
    1 0 1 1;
    0 1 0 1]

A = rand(Bool, 6, 6)
B = rand([0, 0, 0, 0, 1, 2, 3], 10, 15)


@pdf draw_kronecker(A', B) 1500 1000 "poster.pdf"

A = rand([0, 0, 1], 10, 15)
B = collect(0:3)

@pdf draw_kronecker(B, A) 240 600 "rectanle.pdf"


B = [1 2 3; 2 1 3; 3 2 1]'
A = rand([0, 0, 1], 10, 10)

@png draw_kronecker(B', A) 600 600 "logo.png"
