function INP = focusFigure(INP)
%FOCUSFIGURE Focus a figure if not deleted, regenerate window if it was

if exist('INP','var')
    if isgraphics(INP)
        set(0, 'CurrentFigure', INP);
        return
    end
end
INP = figure;

end

